
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

    ### TRAINING DATA
    Shuffle = 1

    ### NET SPECS
    NetSpec = '*FLATFEAT!2-1_*FLATFEAT!2_FC128t_*DP_FC128t_*DP_*ORESHAPE_*LSTM!128_*MASKSEQ_*ADVSPLIT_FC128t'
    AdvSpec = '*GRADFLIP_*DP_FC128t'
    ObservedGrads = '' #separate by _

    # NET TRAINING
    MaxEpochs = 100
    BatchSize = 64 # MULTIPLIED BY 2 (source and target)
    LearnRate = 0.001
    InitStd = 0.1
    EarlyStoppingCondition = 'SOURCEVALID'
    EarlyStoppingPatience = 10

    OutDir = 'TEST.outdir'
    TensorboardDir = './tensorboard'

################################################################################
#################################### SCRIPT ####################################
################################################################################

@ex.automain
def main(
        # Speakers
        SourceSpeakers, TargetSpeakers, ExtraSpeakers, WordsPerSpeaker,
        # Data
        VideoNorm, Shuffle, InitStd,
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
    train_data, _ = data_loader.load_data(Data.SetType.TRAIN, WordsPerSpeaker, VideoNorm, add_channel=False)
    valid_data, _ = data_loader.load_data(Data.SetType.VALID, WordsPerSpeaker, VideoNorm, add_channel=False)
    test_data, feature_size = data_loader.load_data(Data.SetType.TEST, WordsPerSpeaker, VideoNorm, add_channel=False)

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
    builder.add_placeholder(train_source_set.data_dtype, train_source_set.data_shape, 'Sequences')
    builder.add_placeholder(tf.int32, [None], 'SeqLengths')
    builder.add_placeholder(train_source_set.target_dtype, train_source_set.target_shape, 'WordTrgs')
    builder.add_placeholder(train_source_set.domain_dtype, train_source_set.domain_shape, 'DomainTrgs')
    builder.add_placeholder(tf.float32, [], 'Lambda')
    builder.add_placeholder(tf.bool, [], 'Training')

    # Create network
    builder.add_main_specification(NetSpec, 'Sequences', 'WordTrgs')
    builder.add_specification(AdvSpec, 'ADVSPLIT-9/Input', 'DomainTrgs')
    builder.build_model()

    # Training
    stopping_type = Model.StoppingType[EarlyStoppingCondition]
    trainer = Model.AdvTrainer(MaxEpochs, LearnRate, builder.graph_specs, builder.placeholders, TensorboardDir)
    trainer.init_session()
    trainer.train(train_sets=[train_source_set, train_target_set],
                  valid_sets=[valid_source_set, valid_target_set, valid_extra_set],
                  stopping_type=stopping_type,
                  stopping_patience=EarlyStoppingPatience)




