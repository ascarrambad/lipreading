
from . import consts

################################################################################
#################################### WORDS #####################################
################################################################################

__word_encoding = dict()
__word_decoding = dict()
for line in open(consts.WORDLISTFILE, 'r'):
    tup = line.split()
    __word_encoding[tup[1]] = int(tup[0])
    __word_decoding[int(tup[0])] = tup[1]

def word_to_num(s):
    return __word_encoding[s]

def num_to_word(n):
    return __word_decoding[n]

def word_classes_count():
    return max(__word_decoding.keys()) + 1

################################################################################
################################### SPEAKERS ###################################
################################################################################

__spk_encoding = dict()
__spk_decoding = dict()
def encode_speakers(speakers):
    for idx, spk in enumerate(speakers):
        __spk_encoding[spk] = idx
        __spk_decoding[idx] = spk

def speaker_to_num(s):
    return __spk_encoding[s]

def num_to_speaker(n):
    return __spk_decoding[n]

################################################################################
################################## DICTIONARY ##################################
################################################################################

__dictionary = dict()
for line in open(consts.DICTIONARYFILE,'r'):
    line_data = line.split()
    __dictionary[line_data[0]] = line_data[1:]