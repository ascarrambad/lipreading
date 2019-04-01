
import os

import Data

import numpy as np

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
    NetSpec = 'FLATFEAT_FC(OBSA)t128_DP_FC(OBSB)t128_ID(ATTADV)_ID(ATTMVD)_DP_MASKBATCH_LSTMt128'
    AdvSpec = 'FCt100_FCt100'
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
    Remark = ''

################################################################################
#################################### SCRIPT ####################################
################################################################################

@ex.automain
def main(
        # spk and data
        SourceSpeakers,TargetSpeakers,ExtraSpeakers,WordsPerSpeaker,
        # net settings
        BatchSize,MaxEpochs,StopCondition,EarlyStoppingPatience,Shuffle,
        # extra loss settings
        MmdFactors,AdvFactors,
        # rest
        ObservedGrads,OutDir,_config
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
    train_data, _ = data_loader.load_data(train_type, WordsPerSpeaker)
    valid_data, _ = data_loader.load_data(valid_type, WordsPerSpeaker)
    test_data, feature_size = data_loader.load_data(test_type, WordsPerSpeaker)

    # Create source & target datasets for all domain types
    train_source_set = Data.Set(train_data['Source'], BatchSize, Shuffle)
    train_target_set = Data.Set(train_data['Target'], BatchSize, Shuffle)

    valid_source_set = Data.Set(valid_data['Source'], BatchSize, Shuffle)
    valid_target_set = Data.Set(valid_data['Target'], BatchSize, Shuffle)
    valid_extra_set = Data.Set(valid_data['Extra'], BatchSize, Shuffle)

    test_source_set = Data.Set(test_data['Source'], BatchSize, Shuffle)
    test_target_set = Data.Set(test_data['Target'], BatchSize, Shuffle)
    test_extra_set = Data.Set(test_data['Extra'], BatchSize, Shuffle)

    # b1 = train_source_set.next_batch()
    # b2 = train_target_set.next_batch()
    # b3 = b1.concatenate(b2)
    # import pdb; pdb.set_trace()  # breakpoint 6c7267e0 //
    # print('ciao')

    return


