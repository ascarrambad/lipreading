
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

ex = Experiment('LipR.Baseline')

@ex.config
def cfg():

    #### DATA
    AllSpeakers = 's1-s2-s3-s4-s5-s6-s7-s8_s9'
    (SourceSpeakers,TargetSpeakers) = AllSpeakers.split('_')
    WordsPerSpeaker = -1

    ### DATA PROCESSING
    VideoNorm = 'MV'
    AddChannel = False
    DownSample = False

    ### TRAINING DATA
    TruncateRemainder = False
    Shuffle = 1

    ### NET SPECS
    NetSpec = '*FLATFEAT!2-1_*FLATFEAT!2_FC128t_*DP_FC128t_*DP_*ORESHAPE_*LSTM!128_*MASKSEQ_FC128t'
    #

    # NET TRAINING
    MaxEpochs = 200
    BatchSize = 64 # MULTIPLIED BY 2 (source and target)
    LearnRate = 0.0007
    InitStd = 0.1
    EarlyStoppingCondition = 'SOURCEVALID'
    EarlyStoppingValue = 'ACCURACY'
    EarlyStoppingPatience = 10

    DBPath = None
    Collection = 'FC'

    OutDir = 'Outdir/Baseline'
    TensorboardDir = OutDir + '/tensorboard'
    ModelDir = OutDir + '/model'

    # Prepare MongoDB batch exp
    if DBPath != None:
        ex.observers.append(MongoObserver.create(url=DBPath, db_name='LipR_Baseline', collection=Collection))

################################################################################
#################################### SCRIPT ####################################
################################################################################

@ex.automain
def main(
        # Speakers
        AllSpeakers, SourceSpeakers, TargetSpeakers, WordsPerSpeaker,
        # Data
        VideoNorm, AddChannel, DownSample, TruncateRemainder, Shuffle, InitStd,
        # NN settings
        NetSpec,
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
    train_data, _ = data_loader.load_data(Data.SetType.TRAIN, WordsPerSpeaker, VideoNorm, False, AddChannel, DownSample)
    valid_data, _ = data_loader.load_data(Data.SetType.VALID, WordsPerSpeaker, VideoNorm, False, AddChannel, DownSample)
    test_data, feature_size = data_loader.load_data(Data.SetType.TEST, WordsPerSpeaker, VideoNorm, False, AddChannel, DownSample)

    # Create source & target datasets for all domain types
    train_source_set = Data.Set(train_data[Data.DomainType.SOURCE], BatchSize, TruncateRemainder, Shuffle)

    valid_source_set = Data.Set(valid_data[Data.DomainType.SOURCE], BatchSize, TruncateRemainder, Shuffle)
    valid_target_set = Data.Set(valid_data[Data.DomainType.TARGET], BatchSize, TruncateRemainder, Shuffle)

    test_source_set = Data.Set(test_data[Data.DomainType.SOURCE], BatchSize, TruncateRemainder, Shuffle)
    test_target_set = Data.Set(test_data[Data.DomainType.TARGET], BatchSize, TruncateRemainder, Shuffle)

    # Adding classification layers
    NetSpec += '_FC{0}i_*PREDICT!sce'.format(enc.word_classes_count())

    # Model Builder
    builder = Model.Builder(InitStd)

    # Adding placeholders for data
    builder.add_placeholder(train_source_set.data_dtype, train_source_set.data_shape, 'Frames')
    builder.add_placeholder(tf.int32, [None], 'SeqLengths')
    builder.add_placeholder(train_source_set.target_dtype, train_source_set.target_shape, 'WordTrgs')
    builder.add_placeholder(tf.bool, [], 'Training')

    # Create network
    builder.add_specification('NET', NetSpec, 'Frames', 'WordTrgs')
    builder.build_model()

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
                  batch.data_lengths,
                  batch.data_targets,
                  training]

        return dict(zip(keys, values))

    # Training
    stopping_type = Model.StoppingType[EarlyStoppingCondition]
    stopping_value = Model.StoppingValue[EarlyStoppingValue]

    trainer = Model.Trainer(MaxEpochs, optimizer, accuracy, builder.graph_specs[0].loss, losses, TensorboardDir, ModelDir)
    trainer.init_session()
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
