
import os
import re
import itertools

import pickle
import numpy as np

from .Helpers import funcs, consts, encoding, enums
from .Structures.Item import Item
from .Structures.Domain import Domain

class Loader(object):
    def __init__(self, *domain_spk):
        super(Loader, self).__init__()

        if len(domain_spk) == 0:
            raise Exception('Pass some information to init_speaker_list')

        self.domains_speakers = {x[0]: x[1].split('-') for x in domain_spk}

        # Add list with all speakers to be loaded
        all_speakers = [set(x) for x in self.domains_speakers.values()]
        all_speakers = sorted(list(set().union(*all_speakers)))
        self.domains_speakers[enums.DomainType.ALL] = all_speakers

        encoding.encode_words()
        encoding.encode_speakers(all_speakers)

    # Load data from dbtype
    def load_data(self, dbtype, max_words_per_speaker, normalization_vars, diff_frames=False, add_channel=False, downsample=False, verbose=False):

        dmn_spk_tuples = [x for x in self.domains_speakers.items() if x[0] != enums.DomainType.ALL]
        if dbtype == enums.SetType.TRAIN:
            dmn_spk_tuples = [x for x in dmn_spk_tuples if x[0] != enums.DomainType.EXTRA]

        domain_data = {}

        for dmn, spk in dmn_spk_tuples:
            # Get sequence labels
            seq_data = self._collect_seq_data(dbtype, spk, max_words_per_speaker)

            means, stds, diff_means, diff_stds = self._load_video_stats(spk, diff_frames, normalization_vars)
            seq_proc = funcs.sequence_processor(means, stds, add_channel, downsample, diff_frames, diff_means, diff_stds)

            # Load actual data
            binned_data, index_to_bin_pos, feature_size = self._load_and_bin(seq_data, spk, seq_proc, verbose)
            domain_data[dmn] = Domain(dmn, dbtype, binned_data, index_to_bin_pos)

        return domain_data, feature_size

    # return {'speaker:seq': [[word, fromFrame, toFrame]]}
    def _collect_seq_data(self, dbtype, speakers, max_words_per_speaker):

        # one line from db: 'e s1 lwae8n 24750 29250'

        seq_data = {}
        words_per_speaker = { speaker: 0 for speaker in speakers }

        for line in open(dbtype.path):
            if re.match(r'^\s*$',line): # skip empty lines
                continue

            linedata = line.strip().split()
            word = linedata[0]
            speaker = linedata[1]

            if not speaker in speakers:
                continue
            if max_words_per_speaker > 0 and words_per_speaker[speaker] > max_words_per_speaker:
                continue

            words_per_speaker[speaker] += 1

            seqKey = speaker + ':' + linedata[2]
            fromFrame = int(linedata[3]) // 1000  # data is given as 25kHz, downsample for images
            toFrame = int(linedata[4]) // 1000    # to avoid later rounding errors

            if seqKey in seq_data:
                seq_data[seqKey].append((word,fromFrame,toFrame))
            else:
                seq_data[seqKey] = [(word,fromFrame,toFrame)]

        return seq_data

    def _load_video_stats(self, speakers, diff_frames, normalization_vars):
        assert normalization_vars in [ '', 'M', 'MV' ]

        means = {} if 'M' in normalization_vars else None
        stds = {} if 'V' in normalization_vars else None

        diff_means = {} if 'M' in normalization_vars else None
        diff_stds = {} if 'V' in normalization_vars else None

        for spk in speakers:
            if diff_frames:
                diff_means_file = consts.DIFFSTATSDIR + '/means-%s.npy' % spk
                diff_stds_file = consts.DIFFSTATSDIR + '/stds-%s.npy' % spk

            means_file = consts.STATSDIR + '/MEAN-AUD-D.%s-%s.npy' % (consts.VIDEO_INFIX, spk)
            stds_file = consts.STATSDIR + '/STD-AUD-Data.%s-%s.npy' % (consts.VIDEO_INFIX, spk)

            if 'M' in normalization_vars:
                means[spk] = np.load(means_file)
                if diff_frames: diff_means[spk] = np.load(diff_means_file)
            if 'V' in normalization_vars:
                stds[spk] = np.load(stds_file)
                if diff_frames: diff_stds[spk] = np.load(diff_stds_file)

        return means, stds, diff_means, diff_stds

    def _load_and_bin(self, seq_data, speakers, sequence_processor, diff_frames, verbose):

        # load all data as pickle files
        # this is not a major memory hog since we have not yet upsampled (we'll see)
        filename_template = os.path.join(consts.DATADIR,'%(spk)s.Data.%(pp)s.pickle')
        videoData = { spk: pickle.load(open(filename_template % { 'spk': spk, 'pp': consts.VIDEO_INFIX },'rb')) for spk in speakers }

        # mapping from sequence to bin and position (later on used to compute permutation)
        index_to_bin_pos = []

        # a dictionary length bin -> list of Items (length means MAXIMAL length)
        # adapt the keys according to task
        binned_data = { 2: [], 4: [], 6: [], 8: [], 12: [],  -1: [] }

        feature_size = None

        for seqKey in seq_data.keys():
            # Process sequence
            (speaker, name) = seqKey.split(':')
            try:
                seq_pair = videoData[speaker][name]
                (sequence, illegal_frames) = seq_pair
                sequence, diff_sequence = sequence_processor(sequence, speaker)
            except Exception as e:
                if verbose:
                    print('Error for sequence %s: %s' % (seqKey,str(e)))
                sequence = None
                illegal_frames = list(range(consts.TOTAL_MAX_FRAME))

            if sequence is None:
                if verbose:
                    print('Ignoring %s: broken sequence %s' % (word_frames, seqKey))
                continue

            words_frames = seq_data[seqKey] # [(word, fromFrame, toFrame)]

            # Iterate over words and fill return objects
            for word_frames in words_frames:
                (word, fromFrame, toFrame) = word_frames
                rightDelta = (toFrame + 1) - sequence.shape[0]

                if rightDelta > 0:
                    if verbose:
                        print('VIDEO OVERFLOW for %s: %d' % (seqKey, rightDelta))
                    toFrame -= rightDelta
                seq_length = (toFrame + 1) - fromFrame

                if seq_length <= 0 or funcs.any_element_in_range(illegal_frames, fromFrame, toFrame):
                    if verbose:
                        print('SKIPPING VIDEO word %s in %s due to illegal frames' % (word, seqKey))
                    continue

                # Collect data item [key = seqKey + ':' + word]
                if diff_frames:
                    data = diff_sequence[fromFrame:fromFrame + seq_length]
                else:
                    data = sequence[fromFrame:fromFrame + seq_length]
                data_opt = sequence[fromFrame:fromFrame + seq_length] if diff_frames else None
                data_length = np.array(seq_length)
                data_target = encoding.word_one_hot(word)
                domain_target = encoding.speaker_one_hot(speaker)

                item = Item(data=data,
                            data_opt=data_opt,
                            data_length=data_length,
                            data_target=data_target,
                            domain_target=domain_target)

                funcs.to_bin(item, binned_data, index_to_bin_pos)

                if feature_size is None:
                    feature_size = sequence.shape[1:]

        return binned_data, index_to_bin_pos, feature_size
