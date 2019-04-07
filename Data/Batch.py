
import numpy as np

from .Helpers import funcs

class Batch(object):
    """docstring for Batch"""
    def __init__(self, data, data_masks, data_targets, domain_targets):
        super(Batch, self).__init__()

        self.data = data
        self.data_masks = data_masks
        self.data_targets = data_targets
        self.domain_targets = domain_targets

    def concatenate(self, oth_batch):

        # Arrays to pad
        self_array = [self.data, self.data_masks, self.data_targets, self.domain_targets]
        oth_batch_array = [oth_batch.data, oth_batch.data_masks, oth_batch.data_targets, oth_batch.domain_targets]

        arrays_to_be_padded = [self_array, oth_batch_array]

        # Paddings
        max_seq_len = max(self.data.shape[1], oth_batch.data.shape[1])

        paddings = [[[[0, 0], [0, max_seq_len-self.data.shape[1]]]]*4,
                    [[[0, 0], [0, max_seq_len-oth_batch.data.shape[1]]]]*4]
        paddings[0][0] = paddings[0][0] + [[0, 0]] * (len(self.data.shape)-2)
        paddings[1][0] = paddings[1][0] + [[0, 0]] * (len(oth_batch.data.shape)-2)

        # Actual padding
        pad_self_array, pad_oth_batch_array = funcs.pad_nparrays(paddings, arrays_to_be_padded)

        # Concatenate arrays & create new concatenated batch
        concat_arrays = []
        for sarr, barr in zip(pad_self_array, pad_oth_batch_array):
            concat_arrays.append(np.vstack([sarr,barr]))

        return Batch(*concat_arrays)

