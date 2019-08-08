
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
    Shuffle = 1

    ### NET SPECS
    MotSpec = '*FLATFEAT!2-1_CONV16r!5_*MP!2-2_CONV32r!5_*MP!2-2_CONV64r!7_*MP!2-2_*ORESHAPE_*CONVLSTM!64-3_*MASKSEQ'
    #
    CntSpec = 'CONV16r!3_CONV16r!3_*MP!2-2_CONV32r!3_CONV32r!3_*MP!2-2_CONV64r!3_CONV64r!3_CONV64r!3_*MP!2-2'
    #
    EncSpec = '*CONCAT!3_CONV64r!3_CONV32r!3_CONV64r!3'
    #
    ResSpec = '*RESGEN!3'
    #
    DecSpec = '*UNP!2_*RESGET!2_DECONV64r!3_DECONV64r!3_DECONV32r!3_*UNP!2_*RESGET!1_DECONV32r!3_DECONV16r!3_*UNP!2_*RESGET!0_DECONV16r!3_DECONV1t!3'
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
    Collection = 'AUTOENC'

    OutDir = 'Outdir/MCNet.PreProc'
    TensorboardDir = OutDir + '/tensorboard'
    ModelDir = OutDir + '/model'

    # Prepare MongoDB batch exp
    if DBPath != None:
        ex.observers.append(MongoObserver.create(url=DBPath, db_name='LipR_MCNet_PreProc', collection=Collection))

################################################################################
#################################### SCRIPT ####################################
################################################################################

@ex.automain
def main(
        # Speakers
        Speakers, WordsPerSpeaker,
        # Data
        VideoNorm, AddChannel, Shuffle, InitStd,
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
    train_data, _ = data_loader.load_data(Data.SetType.TRAIN, WordsPerSpeaker, VideoNorm, True, AddChannel)
    valid_data, _ = data_loader.load_data(Data.SetType.VALID, WordsPerSpeaker, VideoNorm, True, AddChannel)
    test_data, feature_size = data_loader.load_data(Data.SetType.TEST, WordsPerSpeaker, VideoNorm, True, AddChannel)

    # Create source & target datasets for all domain types
    train_source_set = Data.Set(train_data[Data.DomainType.SOURCE], BatchSize, Shuffle)

    valid_source_set = Data.Set(valid_data[Data.DomainType.SOURCE], BatchSize, Shuffle)

    test_source_set = Data.Set(test_data[Data.DomainType.SOURCE], BatchSize, Shuffle)

    # Adding classification layers
    DecSpec += '_*PREDICT!img'

    # Model Builder
    builder = Model.Builder(InitStd)

    # Adding placeholders for data
    builder.add_placeholder(train_source_set.data_dtype, train_source_set.data_shape, 'Frames')
    builder.add_placeholder(tf.int32, [None], 'SeqLengths')
    builder.add_placeholder(train_source_set.data_dtype, (None,) + feature_size, 'LastFrame')
    builder.add_placeholder(train_source_set.data_dtype, (None,) + feature_size, 'FrameTrgs')
    builder.add_placeholder(tf.bool, [], 'Training')

    # Create network
    builder.add_specification('MOT', MotSpec, 'Frames', None)
    builder.add_specification('CNT', CntSpec, 'LastFrame', None)
    builder.add_specification('ENC', EncSpec, ['MOT-MASKSEQ-9/Output', 'CNT-MP-9/Output'], None)
    res_inputs = ['MOT-CONV-1/Output', 'MOT-CONV-3/Output','MOT-CONV-5/Output',
                  'CNT-CONV-1/Output', 'CNT-CONV-4/Output', 'CNT-CONV-8/Output']
    builder.add_specification('RES', ResSpec, res_inputs, None)
    builder.add_main_specification('DEC', DecSpec, 'ENC-CONV-3/Output', 'FrameTrgs')

    builder.build_model(build_order=['MOT','CNT','RES','ENC','DEC'])

    # Setup Optimizer, Loss, Accuracy
    optimizer = tf.train.AdamOptimizer(LearnRate)

    ## AllLosses array & JointLoss creation
    losses = np.flip(builder.graph_specs[0].loss)

    ## Losses dictionary
    lkeys = list(reversed(['PLoss', 'GdlLoss', 'ImgLoss']))
    losses = dict(zip(lkeys, losses))

    accuracy = builder.graph_specs[0].accuracy

    # Feed Builder
    def feed_builder(epoch, batch, training):

        for i in range(1,batch.data.shape[1]): # range from diff frame 1 to n-1
            keys = builder.placeholders.values()
            seq_lens = np.minimum(batch.data_lengths-1, [i]*64) # -2 because we're interested in the second to last position

            values = [batch.data[:,:i,:,:,:], # all frame diffs until i
                      seq_lens, # length of the sequences
                      batch.data_opt[np.arange(BatchSize),seq_lens,:,:,:], # current frame min(lengths-1, i)
                      batch.data_opt[np.arange(BatchSize),seq_lens,:,:,:], # frame after min(lengths-1, i)
                      training]

            yield dict(zip(keys, values))

    # Training
    stopping_type = Model.StoppingType[EarlyStoppingCondition]
    stopping_value = Model.StoppingValue[EarlyStoppingValue]

    trainer = Model.Trainer(MaxEpochs, optimizer, accuracy, builder.graph_specs[0].loss[2], losses, TensorboardDir, ModelDir)
    trainer.init_session()
    best_e, best_v = trainer.train(train_sets=[train_source_set],
                                   valid_sets=[valid_source_set],
                                   batched_valid=True,
                                   stopping_type=stopping_type,
                                   stopping_value=stopping_value,
                                   stopping_patience=EarlyStoppingPatience,
                                   feed_builder=feed_builder)

    test_result = trainer.test(test_sets=[test_source_set],
                               feed_builder=feed_builder,
                               batched=True)

    if DBPath != None:
        test_result = list(test_result[Data.SetType.TEST].values())
        return [best_e, best_v], list(test_result[0])
