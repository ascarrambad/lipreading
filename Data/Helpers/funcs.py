
import numpy as np

from . import consts

def sequence_processor(means, stds, diff_frames, add_channel, downsample):
    def processingFunction(wordSeq, speaker):
        # reshape to remain generic
        origShape = wordSeq.shape
        wordSeq.shape = (wordSeq.shape[0], np.prod(wordSeq.shape[1:]))

        if diff_frames:
            prev = wordSeq[:-1]
            next_ = wordSeq[1:]
            wordSeq = next_ - prev
            origShape = (origShape[0]-1, origShape[1])

        if means is not None:
            wordSeq -= means[speaker]
        if stds is not None:
            wordSeq /= stds[speaker]

        wordSeq.shape = origShape

        if downsample:
            wordSeq = wordSeq[:,::2,::2]

        if add_channel:
            wordSeq = wordSeq[...,None]
        return wordSeq

    return processingFunction

def any_element_in_range(element_list,range_from,range_to):
    for el in element_list:
        if el >= range_from and el <= range_to:
            return True
    return False

# put an element into its respective bin, and make a note in sqtb (sequence-to-bin)
# key is the sequence key, item is a Item, bd and sqtb are the binnedData and sequenceToBinAndPos variables
def to_bin(item, bd, sqmap):
    for k in sorted([x for x in bd.keys() if x >= 0]):
        if item.data.shape[0] <= k:
            sqmap.append((k, len(bd[k])))
            bd[k].append(item)
            return
    # put it into "-1" bin
    sqmap.append((-1, len(bd[-1])))
    bd[-1].append(item)

def pad_nparrays(paddings, nparrays):

    if len(paddings) == 1:
        paddings = paddings * len(nparrays)

    padded_arrays = []

    if type(nparrays[0]) is list:
        for pad,list_ in zip(paddings, nparrays):
            if type(pad) is not list:
                pad = [pad]
            padded_arrays.append(pad_nparrays(pad, list_))
    else:
        for pad, arr in zip(paddings, nparrays):
            na = np.pad(arr, pad, mode='constant')
            padded_arrays.append(na)

    return padded_arrays
