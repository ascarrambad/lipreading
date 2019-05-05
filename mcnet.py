
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
    AllSpeakers = 's1_s2'
    (SourceSpeakers,TargetSpeakers) = AllSpeakers.split('_')
    WordsPerSpeaker = -1

    ### DATA PROCESSING
    VideoNorm = 'MV'
    AddChannel = True

    ### TRAINING DATA
    Shuffle = 1

    ### NET SPECS
    DynSpec = '*DIFF_*FLATFEAT!2-1_CONV16r!5-1_*MP!2-2_CONV32r!5-1_*MP!2-2_CONV64r!7-1_*MP!2-2_*ORESHAPE_*CONVLSTM!64-3_*MASKSEQ'
    #
    CntSpec = 'CONV16r!3-1_CONV16r!3-1_*MP!2-2_CONV32r!3-1_CONV32r!3-1_*MP!2-2_CONV64r!3-1_CONV64r!3-1_CONV64r!3-1_*MP!2-2'
    #
    TrgSpec = '*CONCAT!3_CONV64r!3-1_CONV32r!3-1_CONV64r!3-1_*FLATFEAT!3_FC64r'
    #
    ObservedGrads = '' #separate by _

    # NET TRAINING
    MaxEpochs = 100
    BatchSize = 64 # MULTIPLIED BY 2 (source and target)
    LearnRate = 0.001
    InitStd = 0.1
    EarlyStoppingCondition = 'SOURCEVALID'
    EarlyStoppingPatience = 10

    OutDir = 'Outdir/MCNet.VALID'
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
    TrgSpec += '_FC{0}i'.format(enc.word_classes_count())

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
    builder.add_main_specification('EDC', TrgSpec, ['DYN-MASKSEQ-10/Output', 'CNT-MP-9/Output'], 'WordTrgs')

    builder.build_model(build_order=['DYN','CNT','EDC'])

    # Training
    stopping_type = Model.StoppingType[EarlyStoppingCondition]
    trainer = Model.ClassicTrainer(MaxEpochs, LearnRate, builder.graph_specs, builder.placeholders, TensorboardDir)
    trainer.init_session()
    trainer.train(train_set=train_source_set,
                  valid_sets=[valid_source_set, valid_target_set],
                  stopping_type=stopping_type,
                  stopping_patience=EarlyStoppingPatience)




