
from . import consts

################################################################################
#################################### WORDS #####################################
################################################################################

_word_encoding = dict()
_word_decoding = dict()
for line in open(consts.WORDLISTFILE, 'r'):
    tup = line.split()
    _word_encoding[tup[1]] = int(tup[0])
    _word_decoding[int(tup[0])] = tup[1]

def word_to_num(s):
    return _word_encoding[s]

def num_to_word(n):
    return _word_decoding[n]

def word_classes_count():
    return max(_word_decoding.keys()) + 1

################################################################################
################################### SPEAKERS ###################################
################################################################################

_spk_encoding = dict()
_spk_decoding = dict()
def encode_speakers(speakers):
    for idx, spk in enumerate(speakers):
        _spk_encoding[spk] = idx
        _spk_decoding[idx] = spk

def speaker_to_num(s):
    return _spk_encoding[s]

def num_to_speaker(n):
    return _spk_decoding[n]

def speaker_classes_count():
    return max(_spk_decoding.keys()) + 1

################################################################################
################################## DICTIONARY ##################################
################################################################################

_dictionary = dict()
for line in open(consts.DICTIONARYFILE,'r'):
    line_data = line.split()
    _dictionary[line_data[0]] = line_data[1:]