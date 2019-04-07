
import os
import re
import itertools

import pickle
import numpy as np

from .Helpers import funcs, consts, encoding
from .Item import Item
from .Domain import Domain

class Loader(object):
    def __init__(self, *domain_spk):
        super(Loader, self).__init__()

        if len(domain_spk) == 0:
            raise Exception('Pass some information to init_speaker_list')

        self.domains_speakers = {x[0]: x[1].split('-') for x in domain_spk}

        # Add list with all speakers to be loaded
        all_speakers = [set(x) for x in self.domains_speakers.values()]
        all_speakers = sorted(list(set().union(*all_speakers)))
        self.domains_speakers['All'] = all_speakers
        encoding.encode_speakers(all_speakers)

    # Load data from dbtype
    def load_data(self, dbtype, max_words_per_speaker, add_channel=False):

        dmn_spk_tuples = [x for x in self.domains_speakers.items() if x[0] != 'All']
        if dbtype.type == 'train':
            dmn_spk_tuples = [x for x in dmn_spk_tuples if x[0] != 'Extra']

        domain_data = {}

        for dmn, spk in dmn_spk_tuples:
            # Get sequence labels
            seq_data = self._collect_seq_data(dbtype, spk, max_words_per_speaker)

            # Load actual data
            binned_data, index_to_bin_pos, feature_size = self._load_and_bin(seq_data, spk, add_channel)
            domain_data[dmn] = Domain(dmn, binned_data, index_to_bin_pos)

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

    def _load_and_bin(self, seq_data, speakers, add_channel, labelResamplingFactor=1): #sequenceProcessor

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
                # sequence = sequenceProcessor(sequence,speaker)
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
                if add_channel: data = np.reshape(data, data.shape + (1,))

                wordmask = np.concatenate(([False] * (seq_length - 1), [True]), axis=0)
                # phonemask = np.full((seq_length,),True,dtype=bool)
                wordtargets = np.concatenate(([0] * (seq_length - 1), [encoding.word_to_num(word)]), axis=0)
                # phonetargets = createFrameTargetsForWord(word,seq_length)
                speakerlabels = np.full((seq_length,), encoding.speaker_to_num(speaker), dtype=int)

                item = Item(data=data,
                            wordmask=wordmask,
                            #phonemask=phonemask,
                            wordtargets=wordtargets,
                            #phonetargets=phonetargets,
                            speakerlabels=speakerlabels)

                funcs.to_bin(item, binned_data, index_to_bin_pos)

                if feature_size is None:
                    feature_size = sequence.shape[1:]

        return binned_data, index_to_bin_pos, feature_size
