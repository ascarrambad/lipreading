
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

ex = Experiment('Evaluator')

@ex.config
def cfg():

    #### PRETRAINING PARAMS
    TrainedModelDir = 'Outdir/MotCnt'
    OutputLayer = 'TRG-PREDICT-3'
    TrainedModelSeed = 650237723

    #### DATA
    AllSpeakers = 's5_s8'
    TrainOrValidOrTest = 0
    WordsPerSpeaker = -1

    ### DATA PROCESSING
    VideoNorm = 'MV'
    AddChannel = True
    DownSample = False
    LoadMotion = False

    ### TRAINING DATA
    TruncateRemainder = False
    Shuffle = 1

    # NET TRAINING
    BatchSize = 64
    LearnRate = 0.0001

    DBPath = None
    DBName = 'LipR_MotCnt'
    Variant = '_DwnSampled'
    Collection = 'NEXTSTEP' + Variant

    ModelDir = TrainedModelDir + '/model%d' % TrainedModelSeed

    # Prepare MongoDB batch exp
    if DBPath != None:
        ex.observers.append(MongoObserver.create(url=DBPath, db_name=DBName, collection=Collection))

################################################################################
#################################### SCRIPT ####################################
################################################################################

@ex.automain
def main(
        #PreProc
        ModelDir, TrainedModelSeed, OutputLayer,
        # Speakers
        AllSpeakers, TrainOrValidOrTest, WordsPerSpeaker,
        # Data
        VideoNorm, AddChannel, DownSample, LoadMotion, TruncateRemainder, Shuffle,
        # Training settings
        BatchSize,
        # Extra settings
        DBPath, _config
        ):
    print('Config directory is:',_config)

    ###########################################################################

    # Data Loader
    dmn_spk = [(Data.DomainType(i), spk) for i,spk in enumerate(AllSpeakers.split('_')) if spk != '']
    data_loader = Data.Loader(*dmn_spk)

    # Load data
    data, feature_size = data_loader.load_data(Data.SetType(TrainOrValidOrTest), WordsPerSpeaker, VideoNorm, LoadMotion, AddChannel, DownSample)

    # Create source & target datasets for all domain types
    datasets = [Data.Set(data[Data.DomainType(i)], BatchSize, TruncateRemainder, Shuffle) for i in range(len(dmn_spk))]

    # Memory cleanup
    del data_loader, data

    # Model Builder
    builder = Model.Builder(init_std=0.1)
    restorer = builder.restore_model(ModelDir)

    ############################################################################
    ############################################################################
    ############################################################################

    # Setup Loss, Accuracy
    # ploss = builder._get_tensor('DEC-PREDICT-5/PLossAVG')
    # gdlloss = builder._get_tensor('DEC-PREDICT-5/GdlLoss/Output')
    # imgloss = builder._get_tensor('DEC-PREDICT-5/ImgLoss')

    # losses = [imgloss, gdlloss, ploss]
    # lkeys = list(reversed(['PLoss', 'GdlLoss', 'ImgLoss']))
    # losses = dict(zip(lkeys, losses))

    # accuracy = builder._get_tensor('DEC-PREDICT-5/AccuracyAVG')

    ########

    loss = builder._get_tensor(OutputLayer+'/MeanLoss')
    accuracy = builder._get_tensor(OutputLayer+'/Accuracy')
    losses = {'Wrd': loss}

    # Feed Builder
    def feed_builder(epoch, batch, training):

        seq_lens = batch.data_lengths-1

        # keys = builder.placeholders.values()
        # values = [batch.data,
        #           batch.data_opt,
        #           batch.data_opt[:,1:,:,:,:],
        #           seq_lens,
        #           False
        #           ]

        ########

        keys = [v for k,v in builder.placeholders.items() if k != 'TrgFrames']
        values = [batch.data,
                  batch.data_targets,
                  batch.data_lengths,
                  False]
        if LoadMotion: values.insert(1, batch.data_opt)

        return dict(zip(keys, values))

    ############################################################################
    ############################################################################
    ############################################################################

    # Training
    trainer = Model.Trainer(epochs=None,
                            optimizer=None,
                            accuracy=accuracy,
                            eval_losses=losses,
                            tensorboard_path=None,
                            model_path=ModelDir)
    trainer.init_session()

    # Restore Parameters
    restorer.restore(trainer.session, tf.train.latest_checkpoint(ModelDir))

    test_result = trainer.test(test_sets=datasets,
                               feed_builder=feed_builder,
                               batched=True)

    if DBPath != None:
        test_result = list(test_result[Data.SetType(TrainOrValidOrTest)].values())
        return [list(test_result[i]) for i in range(len(dmn_spk))]
