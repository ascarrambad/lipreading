
import Data
import Data.Helpers.encoding as enc

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

################################################################################
#################################### SACRED ####################################
################################################################################

from sacred import Experiment

ex = Experiment('DisplayImages')

@ex.config
def cfg():

    #### DATA
    AllSpeakers = 's5_s8'
    WordsPerSpeaker = -1

    ### DATA PROCESSING
    VideoNorm = 'MV'
    AddChannel = True
    DownSample = False

    ### TRAINING DATA
    TruncateRemainder = False
    Shuffle = 1
    BatchSize = 64

################################################################################
#################################### SCRIPT ####################################
################################################################################

@ex.automain
def main(
        # Speakers
        AllSpeakers, WordsPerSpeaker,
        # Data
        VideoNorm, AddChannel, DownSample,
        TruncateRemainder, Shuffle, BatchSize
        ):

    # Data Loader
    dmn_spk = [(Data.DomainType(i), spk) for i,spk in enumerate(AllSpeakers.split('_')) if spk != '']
    data_loader = Data.Loader(*dmn_spk)

    # Load data
    data, feature_size = data_loader.load_data(Data.SetType(0), WordsPerSpeaker, VideoNorm, True, AddChannel, DownSample)

    # Create source & target datasets for all domain types
    datasets = [Data.Set(data[Data.DomainType(i)], BatchSize, TruncateRemainder, Shuffle) for i in range(len(dmn_spk))]

    batch = datasets[0].all

    # Memory cleanup
    del data_loader, data, datasets

    for bs, seql in zip(range(BatchSize), batch.data_lengths):
        width = float(80*seql+8*(seql-1)+8) / 96.
        height = 1
        fig_c, axs_c = plt.subplots(1, seql, figsize=(width, height))
        fig_m, axs_m = plt.subplots(1, seql, figsize=(width, height))

        for seqi in range(seql):
            # new_im = Image.fromarray(batch.data_opt[bs,seqi])
            # new_im = new_im.convert("L")
            # new_im.save("images/cnt/cnt-%d-%d.png" % (bs, seqi))

            # new_im = Image.fromarray(batch.data[bs,seqi])
            # new_im = new_im.convert("L")
            # new_im.save("images/mot/mot-%d-%d.png" % (bs, seqi))


            axs_c[seqi].imshow(batch.data_opt[bs,seqi], cmap='gray')
            axs_c[seqi].label_outer()
            axs_m[seqi].imshow(batch.data[bs,seqi], cmap='gray')
            axs_m[seqi].label_outer()

        fig_c.savefig("../Outdir/mouths/mat_cnt/cnt-%d.png" % bs)
        fig_m.savefig("../Outdir/mouths/mat_mot/mot-%d.png" % bs)