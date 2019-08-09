
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

ex = Experiment('')

@ex.config
def cfg():

    #### PRETRAINING PARAMS
    TrainedModelDir = 'Outdir/MotCnt'
    TrainedModelSeed = 650237723

    #### DATA
    AllSpeakers = 's5_s8'
    ValidOrTest = 0
    WordsPerSpeaker = -1

    ### DATA PROCESSING
    VideoNorm = 'MV'
    AddChannel = True
    DownSample = False

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

    OutDir = TrainedModelDir
    ModelDir = OutDir + '/model%d' % TrainedModelSeed

    # Prepare MongoDB batch exp
    if DBPath != None:
        ex.observers.append(MongoObserver.create(url=DBPath, db_name=DBName, collection=Collection))

################################################################################
#################################### SCRIPT ####################################
################################################################################

@ex.automain
def main(
        #PreProc
        ModelDir, TrainedModelSeed,
        # Speakers
        AllSpeakers, ValidOrTest, WordsPerSpeaker,
        # Data
        VideoNorm, AddChannel, DownSample, TruncateRemainder, Shuffle,
        # Training settings
        BatchSize,
        # Extra settings
        OutDir, DBPath, _config
        ):
    print('Config directory is:',_config)

    ###########################################################################

    # Data Loader
    dmn_spk = [(Data.DomainType(i), spk) for i,spk in enumerate(AllSpeakers.split('_')) if spk != '']
    data_loader = Data.Loader(*dmn_spk)

    # Load data
    data, feature_size = data_loader.load_data(Data.SetType(ValidOrTest+1), WordsPerSpeaker, VideoNorm, True, AddChannel, DownSample)

    # Create source & target datasets for all domain types
    datasets = [Data.Set(data[Data.DomainType(i)], BatchSize, TruncateRemainder, Shuffle) for i in range(len(dmn_spk))]

    del data_loader
    del data

    # Model Builder
    builder = Model.Builder(init_std=0.1)
    restorer = builder.restore_model(ModelDir)

    ############################################################################
    ############################################################################
    ############################################################################

    # Setup Loss, Accuracy
    # ploss = builder._get_tensor('DEC-PREDICT-9/PLossAVG')
    # gdlloss = builder._get_tensor('DEC-PREDICT-9/GdlLoss/Output')
    # imgloss = builder._get_tensor('DEC-PREDICT-9/ImgLoss')

    # losses = [imgloss, gdlloss, ploss]
    # lkeys = list(reversed(['PLoss', 'GdlLoss', 'ImgLoss']))
    # losses = dict(zip(lkeys, losses))

    # accuracy = builder._get_tensor('DEC-PREDICT-9/AccuracyAVG')

    loss = builder._get_tensor('TRG-PREDICT-3/MeanLoss')
    accuracy = builder._get_tensor('TRG-PREDICT-3/Accuracy')
    losses = {'Wrd': loss}

    # Feed Builder
    def feed_builder(epoch, batch, training):

        keys = builder.placeholders.values()
        values = [batch.data,
                  batch.data_lengths,
                  batch.data_opt,
                  batch.data_targets,
                  training]

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
        test_result = list(test_result[Data.SetType(ValidOrTest+1)].values())
        return list(test_result[0]), list(test_result[1])
