
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

    #### PRETRAINING PARAMS
    TrainedModelSeed = 650237723
    MotInputTensor = 'MOT-CONV-3'

    #### DATA
    AllSpeakers = 's5_s8'
    (SourceSpeakers,TargetSpeakers) = AllSpeakers.split('_')
    WordsPerSpeaker = -1

    ### DATA PROCESSING
    VideoNorm = 'MV'
    AddChannel = True
    DownSample = False

    ### TRAINING DATA
    TruncateRemainder = False
    Shuffle = 1

    ### NET SPECS
    #
    MotTSpec = '*STOPGRAD_*FLATFEAT!3_FC128t_*DP_FC128t_*DP_*RESHAPE_*LSTM!128_*MASKSEQ'
    CntTSpec = '*FLATFEAT!2-1_*FLATFEAT!3_FC128t_*DP_FC128t_*DP_*UNDOFLAT!1_*LSTM!128_*MASKSEQ'
    TrgSpec = '*CONCAT!1_FC128t'
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
    Collection = 'NEXTSTEP' + Variant


    OutDir = 'Outdir/MCNet.Class'
    TensorboardDir = None
    ModelDir = OutDir + '/model'

    DBPath = None

    # Prepare MongoDB batch exp
    if DBPath != None:
        ex.observers.append(MongoObserver.create(url=DBPath, db_name='LipR_MCNet_Class_Valid', collection=Collection))

################################################################################
#################################### SCRIPT ####################################
################################################################################

@ex.automain
def main(
        #PreProc
        TrainedModelSeed,  MotInputTensor,
        # Speakers
        SourceSpeakers, TargetSpeakers, WordsPerSpeaker,
        # Data
        VideoNorm, AddChannel, DownSample, TruncateRemainder, Shuffle, InitStd,
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

    restorer = builder.restore_model('Outdir/MCNet.PreProc/model%d/' % TrainedModelSeed)

    # Adding placeholders for data
    builder.add_placeholder(train_source_set.target_dtype, train_source_set.target_shape, 'TrgWords')
    seq_lens = builder.placeholders['SeqLengths']
    training = builder.add_placeholder(tf.bool, [], 'Training')

    # Create network
    mott = builder.add_specification('MOTT', MotTSpec, MotInputTensor+'/Output', None)
    mott.layers['DP-3'].extra_params['TrainingStatusTensor'] = training
    mott.layers['DP-5'].extra_params['TrainingStatusTensor'] = training
    mott.layers['LSTM-7'].extra_params['SequenceLengthsTensor'] = seq_lens

    cntt = builder.add_specification('CNTT', CntTSpec, 'CntTFrames', None)
    cntt.layers['DP-3'].extra_params['TrainingStatusTensor'] = training
    cntt.layers['DP-5'].extra_params['TrainingStatusTensor'] = training
    cntt.layers['LSTM-7'].extra_params['SequenceLengthsTensor'] = seq_lens

    trg = builder.add_specification('TRG', TrgSpec, ['MOTT-MASKSEQ-8/Output', 'CNTT-MASKSEQ-8/Output'], 'TrgWords')
    builder.build_model()

    # Setup Optimizer
    optimizer = tf.train.AdamOptimizer(LearnRate)

    # Feed Builder
    def feed_builder(epoch, batch, training):

        keys = [v for k,v in builder.placeholders.items() if k != 'TrgFrames']
        values = [batch.data,
                  batch.data_opt,
                  batch.data_lengths,
                  batch.data_targets,
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

    # Restore Parameters
    restorer.restore(trainer.session, tf.train.latest_checkpoint('Outdir/MCNet.PreProc/model%d/' % TrainedModelSeed))

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
