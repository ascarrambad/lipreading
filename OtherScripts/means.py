
import os

import numpy as np
import pickle

data_dir = '/home/mwand/projects/AudioVisual/GRID-old/Corpus/%(spk)s'
spk_file = '%(spk)s.Data.NewMouths-Grayscale-80x40-Type1.pickle'

means_file = '/home/rivama/projects/lipreading/DiffStats/means-s%(spk)s.npy'
stds_file = '/home/rivama/projects/lipreading/DiffStats/stds-s%(spk)s.npy'

speakers = ['s%d'%x for x in range(1, 22)]
filename_template = os.path.join(data_dir, spk_file)

for spk in speakers:
    videoData = pickle.load(open(filename_template % { 'spk': spk } ,'rb'))
    videoData = list([x[0] for x in videoData.values()])

    tmp = []

    for seq in videoData:
        prev = seq[:-1]
        next_ = seq[1:]
        diff_seq = next_ - prev

        for fr in diff_seq:
            tmp.append(fr)

    videoData = np.array(tmp)
    del tmp

    videoData.shape = (videoData.shape[0], np.prod(videoData.shape[1:]))

    means = np.mean(videoData, axis=0)
    stds = np.std(videoData, axis=0)

    np.save(means_file%{ 'spk': spk }, means, allow_pickle=False)
    np.save(stds_file%{ 'spk': spk }, stds, allow_pickle=False)