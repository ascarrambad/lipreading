
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

ex = Experiment('LipR.DualSeq')

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
    MotSpec = '*FLATFEAT!2-1_*FLATFEAT!2_FC128t_*DP_FC128t_*DP_*UNDOFLAT!0'
    #
    CntSpec = '*FLATFEAT!2-1_*FLATFEAT!2_FC128t_*DP_FC32t_*DP_FC128t_*DP_*UNDOFLAT!2'
    #
    TrgSpec = '*CONCAT!2_*LSTM!128_*MASKSEQ_FC128t'
    #

    # NET TRAINING
    MaxEpochs = 200
    BatchSize = 64
    LearnRate = 0.0009
    InitStd = 0.1
    EarlyStoppingCondition = 'SOURCEVALID'
    EarlyStoppingValue = 'ACCURACY'
    EarlyStoppingPatience = 10

    DBPath = None
    Variant = ''
    Collection = 'BttlNk_JointLSTM' + Variant

    OutDir = 'Outdir/DualSeq'
    TensorboardDir = OutDir + '/tensorboard'
    ModelDir = OutDir + '/model'

    # Prepare MongoDB batch exp
    if DBPath != None:
        ex.observers.append(MongoObserver.create(url=DBPath, db_name='LipR_DualSeq_Valid', collection=Collection))

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
        MotSpec, CntSpec, TrgSpec,
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
    train_data, _ = data_loader.load_data(Data.SetType.TRAIN, WordsPerSpeaker, VideoNorm, True, AddChannel, DownSample)
    valid_data, _ = data_loader.load_data(Data.SetType.VALID, WordsPerSpeaker, VideoNorm, True, AddChannel, DownSample)

    # Create source & target datasets for all domain types
    train_source_set = Data.Set(train_data[Data.DomainType.SOURCE], BatchSize, TruncateRemainder, Shuffle)

    valid_source_set = Data.Set(valid_data[Data.DomainType.SOURCE], BatchSize, TruncateRemainder, Shuffle)
    valid_target_set = Data.Set(valid_data[Data.DomainType.TARGET], BatchSize, TruncateRemainder, Shuffle)

    # Memory cleanup
    del data_loader, train_data, valid_data

    # Adding classification layers
    TrgSpec += '_FC{0}i_*PREDICT!sce'.format(enc.word_classes_count())

    # Model Builder
    builder = Model.Builder(InitStd)

    # Adding placeholders for data
    builder.add_placeholder(train_source_set.data_dtype, train_source_set.data_shape, 'MotFrames')
    builder.add_placeholder(train_source_set.data_dtype, train_source_set.data_shape, 'CntFrames')
    builder.add_placeholder(train_source_set.target_dtype, train_source_set.target_shape, 'TrgWords')
    seq_lens = builder.add_placeholder(tf.int32, [None], 'SeqLengths')
    training = builder.add_placeholder(tf.bool, [], 'Training')

    # Create network
    mot = builder.add_specification('MOT', MotSpec, 'MotFrames', None)
    mot.layers['DP-3'].extra_params['TrainingStatusTensor'] = training
    mot.layers['DP-5'].extra_params['TrainingStatusTensor'] = training

    cnt = builder.add_specification('CNT', CntSpec, 'CntFrames', None)
    cnt.layers['DP-3'].extra_params['TrainingStatusTensor'] = training
    cnt.layers['DP-5'].extra_params['TrainingStatusTensor'] = training
    cnt.layers['DP-7'].extra_params['TrainingStatusTensor'] = training

    trg = builder.add_specification('TRG', TrgSpec, ['MOT-UNDOFLAT-6/Output', 'CNT-UNDOFLAT-8/Output'], 'TrgWords')
    trg.layers['LSTM-1'].extra_params['SequenceLengthsTensor'] = seq_lens
    trg.layers['MASKSEQ-2'].extra_params['MaskIndicesTensor'] = seq_lens - 1

    builder.build_model()

    # Setup Optimizer
    optimizer = tf.train.AdamOptimizer(LearnRate)

    # Feed Builder
    def feed_builder(epoch, batch, training):

        keys = builder.placeholders.values()
        values = [batch.data,
                  batch.data_opt,
                  batch.data_targets,
                  batch.data_lengths,
                  training]

        return dict(zip(keys, values))

    # Training
    stopping_type = Model.StoppingType[EarlyStoppingCondition]
    stopping_value = Model.StoppingValue[EarlyStoppingValue]

    trainer = Model.Trainer(epochs=MaxEpochs,
                            optimizer=optimizer,
                            accuracy=trg.accuracy,
                            eval_losses={'Wrd': trg.loss},
                            tensorboard_path=TensorboardDir,
                            model_path=ModelDir)
    trainer.init_session()
    best_e, best_v = trainer.train(train_sets=[train_source_set],
                                   valid_sets=[valid_source_set, valid_target_set],
                                   batched_valid=True,
                                   stopping_type=stopping_type,
                                   stopping_value=stopping_value,
                                   stopping_patience=EarlyStoppingPatience,
                                   feed_builder=feed_builder)

    test_result = trainer.test(test_sets=[valid_source_set, valid_target_set],
                               feed_builder=feed_builder,
                               batched=True)

    if DBPath != None:
        test_result = list(test_result[Data.SetType.VALID].values())
        return best_e, list(test_result[0]), list(test_result[1])
