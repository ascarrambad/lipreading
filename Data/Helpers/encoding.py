
import numpy as np

from . import consts

################################################################################
#################################### WORDS #####################################
################################################################################

_word_encoding = dict()
_word_decoding = dict()
_word_one_hot = None
def encode_words():
    global _word_one_hot

    for line in open(consts.WORDLISTFILE, 'r'):
        tup = line.split()
        _word_encoding[tup[1]] = int(tup[0])
        _word_decoding[int(tup[0])] = tup[1]

    _word_one_hot = np.diag(np.ones(word_classes_count(), dtype=np.int))

def word_to_num(s):
    return _word_encoding[s]

def num_to_word(n):
    return _word_decoding[n]

def word_one_hot(word):
    return _word_one_hot[word_to_num(word)]

def word_classes_count():
    return max(_word_decoding.keys()) + 1

################################################################################
################################### SPEAKERS ###################################
################################################################################

_spk_encoding = dict()
_spk_decoding = dict()
_spk_one_hot = None
def encode_speakers(speakers):
    global _spk_one_hot

    for idx, spk in enumerate(speakers):
        _spk_encoding[spk] = idx
        _spk_decoding[idx] = spk

    _spk_one_hot = np.diag(np.ones(speaker_classes_count(), dtype=np.int))

def speaker_to_num(s):
    return _spk_encoding[s]

def num_to_speaker(n):
    return _spk_decoding[n]

def speaker_one_hot(spk):
    return _spk_one_hot[speaker_to_num(spk)]

def speaker_classes_count():
    return max(_spk_decoding.keys()) + 1

################################################################################
################################## DICTIONARY ##################################
################################################################################

_dictionary = dict()
for line in open(consts.DICTIONARYFILE,'r'):
    line_data = line.split()
    _dictionary[line_data[0]] = line_data[1:]