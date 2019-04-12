
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

        encoding.encode_speakers(all_speakers)

    # Load data from dbtype
    def load_data(self, dbtype, max_words_per_speaker, normalization_vars, add_channel=False):

        dmn_spk_tuples = [x for x in self.domains_speakers.items() if x[0] != enums.DomainType.ALL]
        if dbtype == enums.SetType.TRAIN:
            dmn_spk_tuples = [x for x in dmn_spk_tuples if x[0] != enums.DomainType.EXTRA]

        domain_data = {}

        for dmn, spk in dmn_spk_tuples:
            # Get sequence labels
            seq_data = self._collect_seq_data(dbtype, spk, max_words_per_speaker)

            (means, stds) = self._load_video_stats(spk, normalization_vars)
            seq_proc = funcs.sequence_processor(means, stds, add_channel)

            # Load actual data
            binned_data, index_to_bin_pos, feature_size = self._load_and_bin(seq_data, spk, seq_proc)
            domain_data[dmn] = Domain(dmn, dbtype, binned_data, index_to_bin_pos)

        return domain_data, feature_size

    def _load_video_stats(self, speakers, normalization_vars):
        assert normalization_vars in [ '', 'M', 'MV' ]

        means = {} if 'M' in normalization_vars else None
        stds = {} if 'V' in normalization_vars else None

        for spk in speakers:
            means_file = consts.STATSDIR + '/MEAN-AUD-Data.%s-%s.npy' % (consts.VIDEO_INFIX, spk)
            stds_file = consts.STATSDIR + '/STD-AUD-Data.%s-%s.npy' % (consts.VIDEO_INFIX, spk)

            if 'M' in normalization_vars:
                means[spk] = np.load(means_file)
            if 'V' in normalization_vars:
                stds[spk] = np.load(stds_file)

        return (means, stds)

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

    def _load_and_bin(self, seq_data, speakers, sequence_processor, labelResamplingFactor=1):

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
            # step 1: process sequence
            (speaker, name) = seqKey.split(':')
            try:
                seq_pair = videoData[speaker][name]
                (sequence, illegal_frames) = seq_pair
                sequence = sequence_processor(sequence, speaker)
            except Exception as e:
                print('Error for sequence %s: %s' % (seqKey,str(e)))
                sequence = None
                illegal_frames = list(range(consts.TOTAL_MAX_FRAME))

            words_frames = seq_data[seqKey]

            # iterate over words and fill return objects
            for word_frames in words_frames:
                if sequence is None:
                    print('Ignoring %s: broken sequence %s' % (word_frames, seqKey))
                    continue

                (word, fromFrame, toFrame) = word_frames
                fromFrame = fromFrame * labelResamplingFactor
                toFrame = toFrame * labelResamplingFactor
                rightDelta = (toFrame + 1) - sequence.shape[0]

                if rightDelta > 0:
                    print('VIDEO OVERFLOW for %s: %d' % (seqKey, rightDelta))
                    toFrame -= rightDelta
                seq_length = (toFrame + 1) - fromFrame

                if seq_length <= 0 or funcs.any_element_in_range(illegal_frames, fromFrame, toFrame):
                    print('SKIPPING VIDEO word %s in %s due to illegal frames' % (word, seqKey))
                    continue

                # finally collect data item!
                # key = seqKey + ':' + word

                data = sequence[fromFrame:fromFrame + seq_length]
                data_lengths = np.array(seq_length)

                one_hot = np.diag(np.ones(encoding.word_classes_count(), dtype=np.int))
                data_targets = one_hot[encoding.word_to_num(word)]

                one_hot = np.diag(np.ones(encoding.speaker_classes_count(), dtype=np.int))
                domain_targets = one_hot[encoding.speaker_to_num(speaker)]

                item = Item(data=data,
                            data_lengths=data_lengths,
                            data_targets=data_targets,
                            domain_targets=domain_targets)

                funcs.to_bin(item, binned_data, index_to_bin_pos)

                if feature_size is None:
                    feature_size = sequence.shape[1:]

        return binned_data, index_to_bin_pos, feature_size
