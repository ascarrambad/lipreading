
import os

import Data
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
    # speakers, note: train, cv, split by _, speakers separated by -
#     AllSpeakers = 's1_s1_s1'
#     AllSpeakers = 's1-s2-s3-s4_s1-s2-s3-s4_s1-s2-s3-s4'

    #### DATA
    AllSpeakers = 's1_s2_s3'
    (SourceSpeakers,TargetSpeakers,ExtraSpeakers) = AllSpeakers.split('_')
    WordsPerSpeaker = -1

    ### DATA PROCESSING
    VideoNorm = 'MV'

    ### TRAINING DATA
    Shuffle = 1

    ### NET SPECS
    # NetSpec = '*CONVLSTM!2-8_*DP_CONV16r!2-1_*MP!2-2_*DP_CONV32r!2-1_*MP!2-2_*FLATFEAT!3_FC128t-51t'
    NetSpec = '*FLATFEAT!2-1_CONV32r!2-1_*MP!2-2_CONV64r!2-1_*MP!2-2_*DP_*FLATFEAT!3_*ORESHAPE_LSTM128t!0_FC51t'
    AdvSpec = '*GRADFLIP(Reversal)_FC(DomainFC)64t-3t'
    AdvFactors = '1-1.0'
    ObservedGrads = '' #separate by _

    # NET TRAINING
    MaxEpochs = 100
    BatchSize = 64 # MULTIPLIED BY 2 (source and target)
    LearnRate = 0.001
    InitStd = 0.1
    StopCondition = 'EarlyStopOnSourceDev'
    EarlyStoppingPatience = 30

    OutDir = 'TEST.outdir'
    TensorboardDir = './tensorboard'
    Remark = ''

################################################################################
#################################### SCRIPT ####################################
################################################################################

@ex.automain
def main(
        # Speakers
        SourceSpeakers,TargetSpeakers,ExtraSpeakers,WordsPerSpeaker,
        # Data
        VideoNorm,Shuffle,InitStd,
        # NN settings
        NetSpec,AdvSpec,
        # Training settings
        BatchSize,LearnRate,MaxEpochs,StopCondition,EarlyStoppingPatience,
        # Extra settings
        ObservedGrads,OutDir,TensorboardDir,_config
        ):
    print('Config directory is:',_config)

    ###########################################################################
    # Prepare output directory
    try:
        os.makedirs(OutDir)
    except OSError as e:
        print('Error %s when making output dir - ignoring' % str(e))

    # Data Loader
    data_loader = Data.Loader(('Source', SourceSpeakers),
                            ('Target', TargetSpeakers),
                            ('Extra', ExtraSpeakers))

    # Dataset Specifier
    train_type = Data.SetType('train')
    valid_type = Data.SetType('valid')
    test_type = Data.SetType('test')

    # Load data
    train_data, _ = data_loader.load_data(train_type, WordsPerSpeaker, VideoNorm, add_channel=True)
    valid_data, _ = data_loader.load_data(valid_type, WordsPerSpeaker, VideoNorm, add_channel=True)
    test_data, feature_size = data_loader.load_data(test_type, WordsPerSpeaker, VideoNorm, add_channel=True)

    # Create source & target datasets for all domain types
    train_source_set = Data.Set(train_data['Source'], BatchSize, Shuffle)
    train_target_set = Data.Set(train_data['Target'], BatchSize, Shuffle)

    valid_source_set = Data.Set(valid_data['Source'], BatchSize, Shuffle)
    valid_target_set = Data.Set(valid_data['Target'], BatchSize, Shuffle)
    valid_extra_set = Data.Set(valid_data['Extra'], BatchSize, Shuffle)

    test_source_set = Data.Set(test_data['Source'], BatchSize, Shuffle)
    test_target_set = Data.Set(test_data['Target'], BatchSize, Shuffle)
    test_extra_set = Data.Set(test_data['Extra'], BatchSize, Shuffle)

    # Model Builder
    import pdb; pdb.set_trace()  # breakpoint 0018a623 //
    builder = Model.Builder(InitStd)

    # data_type = tf.float32#tf.as_dtype(train_source_set.data_dtype)
    # data_shape = (64, 50, 80, 40, 1)#train_source_set.data_shape

    # target_type = tf.int64#tf.as_dtype(train_source_set.target_dtype)
    # target_shape = (None)#train_source_set.target_shape

    # Adding placeholders for data
    builder.add_placeholder(train_source_set.data_dtype, train_source_set.data_shape, 'Sequences')
    builder.add_placeholder(tf.int64, [None], 'SeqLengths')
    builder.add_placeholder(train_source_set.target_dtype, train_source_set.target_shape, 'WordTrgs')
    builder.add_placeholder(train_source_set.domain_dtype, train_source_set.domain_shape, 'DomainTrgs')
    builder.add_placeholder(tf.float32, [], 'Lambda')
    builder.add_placeholder(tf.bool, [], 'Training')

    # Create network
    builder.add_main_specification(NetSpec, 'Sequences', 'WordTrgs')
    # builder.add_specification(AdvSpec, 'Adv/Input', 'DomainTrgs')
    import pdb; pdb.set_trace()  # breakpoint 308fdf86 //
    builder.build_model()

    import pdb; pdb.set_trace()  # breakpoint 69079128 //
    trainer = Model.Trainer(MaxEpochs, LearnRate, builder.graph_specs, builder.placeholders, TensorboardDir)
    trainer.init_session()
    trainer.train([train_source_set], [valid_source_set])
    trainer.init_session()
    trainer.test([valid_target_set])

    return


