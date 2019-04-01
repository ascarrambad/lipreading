
import numpy as np

from . import consts

def load_video_stats(speakers, VideoNorm):
    assert VideoNorm in [ '', 'M', 'MV' ]

    means = {} if 'M' in VideoNorm else None
    stds = {} if 'V' in VideoNorm else None

    for spk in speakerList:
        means_file = consts.STATSDIR + '/MEAN-AUD-Data.%s-%s.npy' % (consts.VIDEO_INFIX,spk)
        stds_file = consts.STATSDIR + '/STD-AUD-Data.%s-%s.npy' % (consts.VIDEO_INFIX,spk)

        if 'M' in VideoNorm:
            means[spk] = np.load(means_file)
        if 'V' in VideoNorm:
            stds[spk] = np.load(stds_file)

    return (means, stds)

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
        for pad,llist in zip(paddings, nparrays):
            if type(pad) is not list:
                pad = [pad]
            padded_arrays.append(pad_nparrays(pad, llist))
    else:
        for pad, arr in zip(paddings, nparrays):
            na = np.pad(arr, pad, mode='constant')
            padded_arrays.append(na)

    return padded_arrays
