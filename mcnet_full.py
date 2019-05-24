
import os

import Data
import Data.Helpers.encoding as enc
import Model

import numpy as np
import tensorflow as tf

################################################################################
#################################### SACRED ####################################
################################################################################

import sacred

ex = sacred.Experiment('GRID_MCNet')

@ex.config
def cfg():

    #### DATA
    AllSpeakers = 's1-s2-s3-s4-s5-s6-s7-s8_s9'
    (SourceSpeakers,TargetSpeakers) = AllSpeakers.split('_')
    WordsPerSpeaker = -1

    ### DATA PROCESSING
    VideoNorm = 'MV'
    AddChannel = True

    ### TRAINING DATA
    Shuffle = 1

    #
    DynSpec = '*DIFF_*FLATFEAT!2-1_CONV16r!5_*MP!2-2_CONV32r!5_*MP!2-2_CONV64r!7_*MP!2-2_*ORESHAPE_*CONVLSTM!64-7_*MASKSEQ'
    #
    CntSpec = 'CONV16r!3_CONV16r!3_*MP!2-2_CONV32r!3_CONV32r!3_*MP!2-2_CONV64r!3_CONV64r!3_CONV64r!3_*MP!2-2'
    #
    EncSpec = '*CONCAT!3_CONV64r!3_CONV32r!3_CONV64r!3'
    #
    DecSpec = '*UNP!2_DECONV64r!3_DECONV64r!3_DECONV32r!3_*UNP!2_DECONV32r!3_DECONV16r!3_*UNP!2_DECONV16r!3_DECONV1t!3_*FLATFEAT!3_*PREDICT!mse'
    #

    ObservedGrads = '' #separate by _

    # NET TRAINING
    MaxEpochs = 100
    BatchSize = 64
    LearnRate = 0.001
    InitStd = 0.1
    EarlyStoppingCondition = 'SOURCEVALID'
    EarlyStoppingPatience = 10

    OutDir = 'Outdir/MCNet.FULL.VALID'
    TensorboardDir = OutDir + '/tensorboard'

################################################################################
#################################### SCRIPT ####################################
################################################################################

@ex.automain
def main(
        # Speakers
        SourceSpeakers, TargetSpeakers, WordsPerSpeaker,
        # Data
        VideoNorm, AddChannel, Shuffle, InitStd,
        # NN settings
        DynSpec, CntSpec, EncSpec, DecSpec,
        # Training settings
        BatchSize, LearnRate, MaxEpochs, EarlyStoppingCondition, EarlyStoppingPatience,
        # Extra settings
        ObservedGrads, OutDir, TensorboardDir, _config
        ):
    print('Config directory is:',_config)

    ###########################################################################
    # Prepare output directory
    try:
        os.makedirs(OutDir)
    except OSError as e:
        print('Error %s when making output dir - ignoring' % str(e))

    # Data Loader
    data_loader = Data.Loader((Data.DomainType.SOURCE, SourceSpeakers),
                            (Data.DomainType.TARGET, TargetSpeakers))

    # Load data
    train_data, _ = data_loader.load_data(Data.SetType.TRAIN, WordsPerSpeaker, VideoNorm, AddChannel)
    valid_data, _ = data_loader.load_data(Data.SetType.VALID, WordsPerSpeaker, VideoNorm, AddChannel)
    test_data, feature_size = data_loader.load_data(Data.SetType.TEST, WordsPerSpeaker, VideoNorm, AddChannel)

    # Create source & target datasets for all domain types
    train_source_set = Data.Set(train_data[Data.DomainType.SOURCE], BatchSize, Shuffle)

    valid_source_set = Data.Set(valid_data[Data.DomainType.SOURCE], BatchSize, Shuffle)
    valid_target_set = Data.Set(valid_data[Data.DomainType.TARGET], BatchSize, Shuffle)

    test_source_set = Data.Set(test_data[Data.DomainType.SOURCE], BatchSize, Shuffle)
    test_target_set = Data.Set(test_data[Data.DomainType.TARGET], BatchSize, Shuffle)

    # Adding classification layers
    # DecSpec += '_FC{0}i'.format(enc.word_classes_count())

    # Model Builder
    builder = Model.Builder(InitStd)

    # Adding placeholders for data
    builder.add_placeholder(train_source_set.data_dtype, train_source_set.data_shape, 'Frames')
    builder.add_placeholder(tf.int32, [None], 'SeqLengths')
    builder.add_placeholder(train_source_set.data_dtype, (None,) + feature_size, 'LastFrame')
    builder.add_placeholder(train_source_set.data_dtype, (None,) + (np.prod(feature_size),), 'FrameTrgs')
    builder.add_placeholder(tf.bool, [], 'Training')

    # Create network
    builder.add_specification('DYN', DynSpec, 'Frames', None)
    builder.add_specification('CNT', CntSpec, 'LastFrame', None)
    builder.add_specification('ENC', EncSpec, ['DYN-MASKSEQ-10/Output', 'CNT-MP-9/Output'], None)
    builder.add_main_specification('DEC', DecSpec, 'ENC-CONV-3/Output', 'FrameTrgs')

    builder.build_model(build_order=['DYN','CNT','ENC','DEC'])

    # Setup Optimizer, Loss, Accuracy
    optimizer = tf.train.AdamOptimizer(LearnRate)

    ## AllLosses array & JointLoss creation
    losses = [x.loss for x in builder.graph_specs if x.loss != None]

    ## Losses dictionary
    lkeys = ['Wrd']
    losses = dict(zip(lkeys, losses))

    accuracy = builder.graph_specs[0].accuracy

    # Feed Builder
    def feed_builder(epoch, batch, training):

        keys = builder.placeholders.values()
        values = [batch.data,
                  batch.data_lengths-2,
                  batch.data[np.arange(len(batch.data)),batch.data_lengths-2],
                  batch.data[np.arange(len(batch.data)),batch.data_lengths-1].reshape((BatchSize,-1)),
                  training]

        return dict(zip(keys, values))

    # Training
    stopping_type = Model.StoppingType[EarlyStoppingCondition]
    trainer = Model.Trainer(MaxEpochs, optimizer, accuracy, builder.graph_specs[0].loss, losses, TensorboardDir)
    trainer.init_session()
    trainer.train(train_sets=[train_source_set],
                  valid_sets=[valid_source_set, valid_target_set],
                  batched_valid=True,
                  stopping_type=stopping_type,
                  stopping_patience=EarlyStoppingPatience,
                  feed_builder=feed_builder)




