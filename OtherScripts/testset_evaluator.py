
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

    #### DATA
    Speakers = 's1'
    WordsPerSpeaker = -1

    ### DATA PROCESSING
    VideoNorm = 'MV'
    AddChannel = True
    DownSample = False

    ### TRAINING DATA
    TruncateRemainder = False
    Shuffle = 1

    # NET TRAINING
    MaxEpochs = 200
    BatchSize = 64
    LearnRate = 0.0009
    InitStd = 0.1

    TensorboardDir = None
    ModelDir = None
    DBPath = None

################################################################################
#################################### SCRIPT ####################################
################################################################################

@ex.automain
def main(
        #PreProc
        TrainedModelSeed,
        # Speakers
        Speakers, WordsPerSpeaker,
        # Data
        VideoNorm, AddChannel, DownSample, TruncateRemainder, Shuffle, InitStd,
        # Training settings
        BatchSize, LearnRate, MaxEpochs,
        # Extra settings
        ModelDir, TensorboardDir, DBPath, _config
        ):
    print('Config directory is:',_config)

    ###########################################################################

    # Data Loader
    data_loader = Data.Loader((Data.DomainType.SOURCE, Speakers))

    # Load data
    test_data, feature_size = data_loader.load_data(Data.SetType.TEST, WordsPerSpeaker, VideoNorm, True, AddChannel, DownSample)

    # Create source & target datasets for all domain types
    test_source_set = Data.Set(test_data[Data.DomainType.SOURCE], BatchSize, TruncateRemainder, Shuffle)

    # Model Builder
    builder = Model.Builder(InitStd)

    restorer = builder.restore_model('Outdir/MCNet.PreProc/model%d/' % TrainedModelSeed)

    # Setup Loss, Accuracy
    ploss = builder._get_tensor('DEC-PREDICT-9/MeanLoss')
    gdlloss = builder._get_tensor('DEC-PREDICT-9/ImgLoss/GdlLoss/GdlLoss')
    imgloss = builder._get_tensor('DEC-PREDICT-9/ImgLoss/ImgLoss')

    losses = [imgloss, gdlloss, ploss]
    lkeys = list(reversed(['PLoss', 'GdlLoss', 'ImgLoss']))
    losses = dict(zip(lkeys, losses))

    accuracy = builder._get_tensor('DEC-PREDICT-9/MeanAccuracy')

    # Feed Builder
    def feed_builder(epoch, batch, training):

        batch_size = batch.data.shape[0]
        seq_lens = batch.data_lengths-1 # -1 because we're interested in the second to last position (last position must be predicted)

        keys = builder.placeholders.values()
        values = [batch.data,
                  seq_lens,
                  batch.data_opt,
                  batch.data_opt[:,1:,:,:,:]]

        return dict(zip(keys, values))

    # Training
    trainer = Model.Trainer(epochs=MaxEpochs,
                            optimizer=None,
                            accuracy=accuracy,
                            eval_losses=losses,
                            tensorboard_path=TensorboardDir,
                            model_path=ModelDir)
    trainer.init_session()

    # Restore Parameters
    restorer.restore(trainer.session, tf.train.latest_checkpoint('Outdir/MCNet.PreProc/model%d/' % TrainedModelSeed))

    test_result = trainer.test(test_sets=[test_source_set],
                               feed_builder=feed_builder,
                               batched=True)

