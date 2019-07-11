
import os
import sys

import Data
import Data.Helpers.encoding as enc
import Model

import numpy as np
import tensorflow as tf

################################################################################
#################################### SACRED ####################################
################################################################################

from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment('LipR.MCNet.Class')

@ex.config
def cfg():

    #### DATA
    TrainedModelSeed = 650237723 #800430375
    AllSpeakers = 's5_s8'
    (SourceSpeakers,TargetSpeakers) = AllSpeakers.split('_')
    WordsPerSpeaker = -1

    ### DATA PROCESSING
    VideoNorm = 'MV'
    AddChannel = True

    ### TRAINING DATA
    Shuffle = 1

    ### NET SPECS
    #
    MotTSpec = '*STOPGRAD_*FLATFEAT!2-1_*FLATFEAT!3_FC128t_*DP_FC128t_*DP_*ORESHAPE_*LSTM!128_*MASKSEQ'
    CntTSpec = '*STOPGRAD_*FLATFEAT!3'
    TrgSpec = '*CONCAT!1_FC128t'
    #

    # NET TRAINING
    MaxEpochs = 200
    BatchSize = 64
    LearnRate = 0.0001
    InitStd = 0.1
    EarlyStoppingCondition = 'SOURCEVALID'
    EarlyStoppingValue = 'ACCURACY'
    EarlyStoppingPatience = 10

    DBPath = None
    Collection = 'NEXTSTEP'

    OutDir = 'Outdir/MCNet.Class'
    TensorboardDir = None
    ModelDir = OutDir + '/model'

    DBPath = None

    # Prepare MongoDB batch exp
    if DBPath != None:
        ex.observers.append(MongoObserver.create(url=DBPath, db_name='LipR_MCNet_Class', collection=Collection))

################################################################################
#################################### SCRIPT ####################################
################################################################################

@ex.automain
def main(
        # Speakers
        TrainedModelSeed, SourceSpeakers, TargetSpeakers, WordsPerSpeaker,
        # Data
        VideoNorm, AddChannel, Shuffle, InitStd,
        # NN settings
        MotTSpec, CntTSpec, TrgSpec,
        # Training settings
        BatchSize, LearnRate, MaxEpochs, EarlyStoppingCondition, EarlyStoppingValue, EarlyStoppingPatience,
        # Extra settings
        OutDir, ModelDir, TensorboardDir, DBPath, _config
        ):
    print('Config directory is:',_config)

    ###########################################################################
    # Prepare output directory
    try:
        os.makedirs(OutDir)
    except OSError as e:
        print('Error %s when making output dir - ignoring' % str(e))

    if TensorboardDir is not None:
        TensorboardDir = TensorboardDir + '%d' % _config['seed']
    if ModelDir is not None:
        ModelDir = ModelDir + '%d' % _config['seed']

    if DBPath != None:
        LogPath = OutDir + '/Logs/%d.txt' % _config['seed']

        try: os.makedirs(os.path.dirname(LogPath))
        except OSError as exc: pass

        sys.stdout = open(LogPath, 'w+')

    # Data Loader
    data_loader = Data.Loader((Data.DomainType.SOURCE, SourceSpeakers),
                              (Data.DomainType.TARGET, TargetSpeakers))

    # Load data
    train_data, _ = data_loader.load_data(Data.SetType.TRAIN, WordsPerSpeaker, VideoNorm, True, AddChannel)
    valid_data, _ = data_loader.load_data(Data.SetType.VALID, WordsPerSpeaker, VideoNorm, True, AddChannel)
    test_data, feature_size = data_loader.load_data(Data.SetType.TEST, WordsPerSpeaker, VideoNorm, True, AddChannel)

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

    restorer = builder.restore_model('Outdir/MCNet.PreProc/model%d/' % TrainedModelSeed)

    # Adding placeholders for data
    builder.add_placeholder(train_source_set.target_dtype, train_source_set.target_shape, 'WordTrgs')

    # Create network
    builder.add_specification('MOTT', MotTSpec, 'DYN-ORESHAPE-7/Output', None)
    builder.add_specification('CNTT', CntTSpec, 'CNT-MP-9/Output', None)
    builder.add_main_specification('TRG', TrgSpec, ['MOTT-MASKSEQ-9/Output', 'CNTT-FLATFEAT-1/Output'], 'WordTrgs')
    builder.build_model(build_order=['MOTT', 'CNTT', 'TRG'])

    # Setup Optimizer, Loss, Accuracy
    optimizer = tf.train.AdamOptimizer(LearnRate)

    ## AllLosses array & JointLoss creation
    losses = [builder.graph_specs[0].loss]

    ## Losses dictionary
    lkeys = ['Loss']
    losses = dict(zip(lkeys, losses))

    accuracy = builder.graph_specs[0].accuracy

    # Feed Builder
    def feed_builder(epoch, batch, training):

        keys = [v for k,v in builder.placeholders.items() if k != 'FrameTrgs']
        values = [batch.data,
                  batch.data_lengths,
                  batch.data_opt[np.arange(BatchSize),batch.data_lengths-1,:,:,:],
                  training,
                  batch.data_targets]

        return dict(zip(keys, values))

    # Training
    stopping_type = Model.StoppingType[EarlyStoppingCondition]
    stopping_value = Model.StoppingValue[EarlyStoppingValue]

    trainer = Model.Trainer(MaxEpochs, optimizer, accuracy, builder.graph_specs[0].loss, losses, TensorboardDir, ModelDir)
    trainer.init_session()

    # Restore Parameters
    restorer.restore(trainer.session, tf.train.latest_checkpoint('Outdir/MCNet.NEXTSTEP/model%d/' % TrainedModelSeed))

    best_e, best_v = trainer.train(train_sets=[train_source_set],
                                   valid_sets=[valid_source_set, valid_target_set],
                                   batched_valid=True,
                                   stopping_type=stopping_type,
                                   stopping_value=stopping_value,
                                   stopping_patience=EarlyStoppingPatience,
                                   feed_builder=feed_builder)

    test_result = trainer.test(test_sets=[test_source_set, test_target_set],
                               feed_builder=feed_builder,
                               batched=True)

    if DBPath != None:
        test_result = list(test_result[Data.SetType.TEST].values())
        return [best_e, best_v], list(test_result[0]), list(test_result[1])
