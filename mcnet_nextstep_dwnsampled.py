
import os
import sys

import Data
import Data.Helpers.encoding as enc
import Model

import numpy as np
import tensorflow as tf

from CustomLayers.imgloss import imgloss

################################################################################
#################################### SACRED ####################################
################################################################################

from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment('LipR.MCNet.PreProc')

@ex.config
def cfg():

    #### DATA
    Speakers = 's1-s2-s3-s4-s5'
    WordsPerSpeaker = -1

    ### DATA PROCESSING
    VideoNorm = 'MV'
    AddChannel = True

    ### TRAINING DATA
    TruncateRemainder = False
    Shuffle = 1

    ### NET SPECS
    MotSpec = '*FLATFEAT!2-1_CONV32r!5_*MP!2-2_CONV64r!5_*MP!2-2_*UNDOFLAT!0_*CONVLSTM!64-3'
    #
    CntSpec = '*FLATFEAT!2-1_CONV32r!3_CONV32r!3_*MP!2-2_CONV64r!3_CONV64r!3_*MP!2-2_*UNDOFLAT!1_*CONVLSTM!64-3'
    #
    EncSpec = '*CONCAT!4_*FLATFEAT!2-1_CONV64r!3_CONV32r!3_CONV64r!3'
    #
    ResSpec = '*RESGEN!3'
    #
    DecSpec = '*UNP!2_*RESGET!1_DECONV64r!3_DECONV32r!3_*UNP!2_*RESGET!0_DECONV32r!3_DECONV1t!3_*UNDOFLAT!2'
    #

    # NET TRAINING
    MaxEpochs = 200
    BatchSize = 64
    LearnRate = 0.0001
    InitStd = 0.1
    EarlyStoppingCondition = 'SOURCEVALID'
    EarlyStoppingValue = 'LOSS'
    EarlyStoppingPatience = 10

    DBPath = None
    Variant = '_DwnSampled'
    Collection = 'NEXTSTEP' + Variant

    OutDir = 'Outdir/MCNet.PreProc'
    TensorboardDir = OutDir + '/tensorboard'
    ModelDir = OutDir + '/model'

    # Prepare MongoDB batch exp
    if DBPath != None:
        ex.observers.append(MongoObserver.create(url=DBPath, db_name='LipR_MCNet_PreProc_Valid', collection=Collection))

################################################################################
#################################### SCRIPT ####################################
################################################################################

@ex.automain
def main(
        # Speakers
        Speakers, WordsPerSpeaker,
        # Data
        VideoNorm, AddChannel, TruncateRemainder, Shuffle, InitStd,
        # NN settings
        MotSpec, CntSpec, EncSpec, ResSpec, DecSpec,
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
    data_loader = Data.Loader((Data.DomainType.SOURCE, Speakers))

    # Load data
    train_data, _ = data_loader.load_data(Data.SetType.TRAIN, WordsPerSpeaker, VideoNorm, True, AddChannel, downsample=True)
    valid_data, _ = data_loader.load_data(Data.SetType.VALID, WordsPerSpeaker, VideoNorm, True, AddChannel, downsample=True)

    # Create source & target datasets for all domain types
    train_source_set = Data.Set(train_data[Data.DomainType.SOURCE], BatchSize, TruncateRemainder, Shuffle)
    valid_source_set = Data.Set(valid_data[Data.DomainType.SOURCE], BatchSize, TruncateRemainder, Shuffle)

    # Memory cleanup
    del data_loader, train_data, valid_data

    # Adding classification layers
    DecSpec += '_*CUSTOM(PREDICT)'

    # Model Builder
    builder = Model.Builder(InitStd)

    # Adding placeholders for data
    builder.add_placeholder(train_source_set.data_dtype, train_source_set.data_shape, 'MotFrames')
    builder.add_placeholder(train_source_set.data_dtype, train_source_set.data_shape, 'CntFrames')
    builder.add_placeholder(train_source_set.data_dtype, train_source_set.data_shape, 'TrgFrames')
    seq_lens = builder.add_placeholder(tf.int32, [None], 'SeqLengths')

    # Create network
    mot = builder.add_specification('MOT', MotSpec, 'MotFrames', None)
    mot.layers['CONVLSTM-6'].extra_params['SequenceLengthsTensor'] = seq_lens

    cnt = builder.add_specification('CNT', CntSpec, 'CntFrames', None)
    cnt.layers['CONVLSTM-8'].extra_params['SequenceLengthsTensor'] = seq_lens

    res_inputs = ['MOT-CONV-1/Output', 'MOT-CONV-3/Output',
                  'CNT-CONV-2/Output', 'CNT-CONV-5/Output']
    builder.add_specification('RES', ResSpec, res_inputs, None)

    builder.add_specification('ENC', EncSpec, ['MOT-CONVLSTM-6/Output', 'CNT-CONVLSTM-8/Output'], None)

    dec = builder.add_specification('DEC', DecSpec, 'ENC-CONV-4/Output', 'TrgFrames')
    dec.layers['PREDICT-9'].extra_params['CustomFunction'] = imgloss

    builder.build_model()

    # Setup Optimizer, Loss
    optimizer = tf.train.AdamOptimizer(LearnRate)

    ## Losses dictionary
    losses = np.flip(dec.loss)
    lkeys = list(reversed(['PLoss', 'GdlLoss', 'ImgLoss']))
    losses = dict(zip(lkeys, losses))

    # Feed Builder
    def feed_builder(epoch, batch, training):

        # # Padding
        # max_seq_len = max(batch.data.shape[1], batch.data_opt.shape[1])
        # paddings = [[[0, 0], [0, max_seq_len-batch.data.shape[1]]] + [[0, 0]] * (len(batch.data.shape)-2)]
        # [pad_data] = fns.pad_nparrays(paddings, [batch.data])

        seq_lens = batch.data_lengths-1 # -1 because we're interested in the second to last position (last position must be predicted)

        keys = builder.placeholders.values()
        values = [batch.data,
                  batch.data_opt,
                  batch.data_opt[:,1:,:,:,:],
                  seq_lens]

        return dict(zip(keys, values))


    # Training
    stopping_type = Model.StoppingType[EarlyStoppingCondition]
    stopping_value = Model.StoppingValue[EarlyStoppingValue]

    trainer = Model.Trainer(epochs=MaxEpochs,
                            optimizer=optimizer,
                            accuracy=dec.accuracy,
                            eval_losses=losses,
                            tensorboard_path=TensorboardDir,
                            model_path=ModelDir)
    trainer.init_session()
    best_e, best_v = trainer.train(train_sets=[train_source_set],
                                   valid_sets=[valid_source_set],
                                   batched_valid=True,
                                   stopping_type=stopping_type,
                                   stopping_value=stopping_value,
                                   stopping_patience=EarlyStoppingPatience,
                                   feed_builder=feed_builder)

    test_result = trainer.test(test_sets=[valid_source_set],
                               feed_builder=feed_builder,
                               batched=True)

    if DBPath != None:
        test_result = list(test_result[Data.SetType.VALID].values())
        return best_e, list(test_result[0][:-1])
