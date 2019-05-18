
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

ex = sacred.Experiment('GRID_Adversarial')

@ex.config
def cfg():

    #### DATA
    AllSpeakers = 's1-s2-s3_s4-s5-s6_s7-s8-s9'
    (SourceSpeakers,TargetSpeakers,ExtraSpeakers) = AllSpeakers.split('_')
    WordsPerSpeaker = -1

    ### DATA PROCESSING
    VideoNorm = 'MV'
    AddChannel = True

    ### TRAINING DATA
    Shuffle = 1

    ### NET SPECS
    NetSpec = '*FLATFEAT!2-1_CONV32r!5_*MP!2-2_*DP_CONV48r!5_*MP!2-2_*DP_*ORESHAPE_*CONVLSTM!64-5_*MASKSEQ_*FLATFEAT!3_*ADVSPLIT_FC100r_FC100r'
    AdvSpec = '*GRADFLIP_FC100r'
    ObservedGrads = '' #separate by _

    # NET TRAINING
    MaxEpochs = 100
    BatchSize = 64 # MULTIPLIED BY 2 (source and target)
    LearnRate = 0.001
    InitStd = 0.1
    EarlyStoppingCondition = 'SOURCEVALID'
    EarlyStoppingPatience = 10

    OutDir = 'Outdir/ADV.VALID'
    TensorboardDir = OutDir + '/tensorboard'

################################################################################
#################################### SCRIPT ####################################
################################################################################

@ex.automain
def main(
        # Speakers
        SourceSpeakers, TargetSpeakers, ExtraSpeakers, WordsPerSpeaker,
        # Data
        VideoNorm, AddChannel, Shuffle, InitStd,
        # NN settings
        NetSpec, AdvSpec,
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
                            (Data.DomainType.TARGET, TargetSpeakers),
                            (Data.DomainType.EXTRA, ExtraSpeakers))

    # Load data
    train_data, _ = data_loader.load_data(Data.SetType.TRAIN, WordsPerSpeaker, VideoNorm, AddChannel)
    valid_data, _ = data_loader.load_data(Data.SetType.VALID, WordsPerSpeaker, VideoNorm, AddChannel)
    test_data, feature_size = data_loader.load_data(Data.SetType.TEST, WordsPerSpeaker, VideoNorm, AddChannel)

    # Create source & target datasets for all domain types
    train_source_set = Data.Set(train_data[Data.DomainType.SOURCE], BatchSize, Shuffle)
    train_target_set = Data.Set(train_data[Data.DomainType.TARGET], BatchSize, Shuffle)

    valid_source_set = Data.Set(valid_data[Data.DomainType.SOURCE], BatchSize, Shuffle)
    valid_target_set = Data.Set(valid_data[Data.DomainType.TARGET], BatchSize, Shuffle)
    valid_extra_set = Data.Set(valid_data[Data.DomainType.EXTRA], BatchSize, Shuffle)

    test_source_set = Data.Set(test_data[Data.DomainType.SOURCE], BatchSize, Shuffle)
    test_target_set = Data.Set(test_data[Data.DomainType.TARGET], BatchSize, Shuffle)
    test_extra_set = Data.Set(test_data[Data.DomainType.EXTRA], BatchSize, Shuffle)

    # Adding classification layers
    NetSpec += '_FC{0}i'.format(enc.word_classes_count())
    AdvSpec += '_FC{0}i'.format(enc.speaker_classes_count())

    # Model Builder
    builder = Model.Builder(InitStd)

    # Adding placeholders for data
    builder.add_placeholder(train_source_set.data_dtype, train_source_set.data_shape, 'Frames')
    builder.add_placeholder(tf.int32, [None], 'SeqLengths')
    builder.add_placeholder(train_source_set.target_dtype, train_source_set.target_shape, 'WordTrgs')
    builder.add_placeholder(train_source_set.domain_dtype, train_source_set.domain_shape, 'DomainTrgs')
    builder.add_placeholder(tf.float32, [], 'Lambda')
    builder.add_placeholder(tf.bool, [], 'Training')

    # Create network
    builder.add_main_specification('WRD', NetSpec, 'Frames', 'WordTrgs')
    builder.add_specification('SPK', AdvSpec, 'WRD-ADVSPLIT-11/Input', 'DomainTrgs')
    builder.build_model()

    # Setup Optimizer, Loss, Accuracy
    optimizer = tf.train.AdamOptimizer(LearnRate)

    ## AllLosses array & JointLoss creation
    losses = [x.loss for x in builder.graph_specs if x.loss != None]
    jloss = tf.identity(sum(losses), name='JointLoss')
    losses.append(jloss)

    tf.summary.scalar('JointLoss', jloss)

    ## Losses dictionary
    lkeys = ['Seq', 'Adv', 'Joint']
    losses = dict(zip(lkeys, losses))

    accuracy = builder.graph_specs[0].accuracy

    # Feed Builder
    def feed_builder(epoch, batch, training):

        p = float(epoch) / MaxEpochs
        lambda_ = 2. / (1. + np.exp(-10. * p)) - 1

        keys = builder.placeholders.values()
        values = [batch.data,
                  batch.data_lengths,
                  batch.data_targets,
                  batch.domain_targets,
                  lambda_,
                  training]

        return dict(zip(keys, values))

    # Training
    stopping_type = Model.StoppingType[EarlyStoppingCondition]
    trainer = Model.Trainer(MaxEpochs, optimizer, accuracy, jloss, losses, TensorboardDir)
    trainer.init_session()
    trainer.train(train_sets=[train_source_set, train_target_set],
                  valid_sets=[valid_source_set, valid_target_set, valid_extra_set],
                  stopping_type=stopping_type,
                  stopping_patience=EarlyStoppingPatience,
                  feed_builder=feed_builder)




