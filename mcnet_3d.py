
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

ex = sacred.Experiment('GRID_MCNet_CONV3D')

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

    ### NET SPECS
    # DynSpec = 'CONVTD64r!5-2_*MPTD!2-2_CONVTD128r!5-2_*MPTD!2-2_CONVTD256r!7-2_*MPTD!2-2_*CONVLSTM!256-7_*MASKSEQ'
    # #
    # CntSpec = 'CONV64r!3_CONV64r!3_*MP!2-2_CONV128r!3_CONV128r!3_*MP!2-2_CONV256r!3_CONV256r!3_CONV256r!3_*MP!2-2'
    # #
    # TrgSpec = '*CONCAT!3_CONV256r!3-1_CONV128r!3-1_CONV256r!3-1_*FLATFEAT!3_FC256r'

    # DynSpec = 'CONVTD32r!5-2_*MPTD!2-2_CONVTD64r!5-2_*MPTD!2-2_CONVTD128r!7-2_*MPTD!2-2_*CONVLSTM!128-7_*MASKSEQ'
    # #
    # CntSpec = 'CONV32r!3_CONV32r!3_*MP!2-2_CONV64r!3_CONV64r!3_*MP!2-2_CONV128r!3_CONV128r!3_CONV128r!3_*MP!2-2'
    # #
    # TrgSpec = '*CONCAT!3_CONV128r!3_CONV64r!3_CONV128r!3_*FLATFEAT!3_FC128r'

    DynSpec = 'CONVTD16r!5-2_*MPTD!2-2_CONVTD32r!5-2_*MPTD!2-2_CONVTD64r!7-2_*MPTD!2-2_*CONVLSTM!64-7_*MASKSEQ'
    #
    CntSpec = 'CONV16r!3_CONV16r!3_*MP!2-2_CONV32r!3_CONV32r!3-1_*MP!2-2_CONV64r!3_CONV64r!3_CONV64r!3_*MP!2-2'
    #
    TrgSpec = '*CONCAT!3_CONV64r!3_CONV32r!3_CONV64r!3_*FLATFEAT!3_FC64r'
    #
    ObservedGrads = '' #separate by _

    # NET TRAINING
    MaxEpochs = 100
    BatchSize = 64
    LearnRate = 0.001
    InitStd = 0.1
    EarlyStoppingCondition = 'SOURCEVALID'
    EarlyStoppingPatience = 10

    OutDir = 'Outdir/MCNet.3D.VALID'
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
        DynSpec, CntSpec, TrgSpec,
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
    TrgSpec += '_FC{0}i_*PREDICT!sce'.format(enc.word_classes_count())

    # Model Builder
    builder = Model.Builder(InitStd)

    # Adding placeholders for data
    builder.add_placeholder(train_source_set.data_dtype, train_source_set.data_shape, 'Frames')
    builder.add_placeholder(tf.int32, [None], 'SeqLengths')
    builder.add_placeholder(train_source_set.data_dtype, (None,) + feature_size, 'LastFrame')
    builder.add_placeholder(train_source_set.target_dtype, train_source_set.target_shape, 'WordTrgs')
    builder.add_placeholder(tf.bool, [], 'Training')

    # Create network
    builder.add_specification('DYN', DynSpec, 'Frames', None)
    builder.add_specification('CNT', CntSpec, 'LastFrame', None)
    builder.add_main_specification('EDC', TrgSpec, ['DYN-MASKSEQ-7/Output', 'CNT-MP-9/Output'], 'WordTrgs')

    builder.build_model(build_order=['DYN','CNT','EDC'])

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
                  batch.data_lengths-1,
                  batch.data[np.arange(len(batch.data)),batch.data_lengths-1],
                  batch.data_targets,
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



