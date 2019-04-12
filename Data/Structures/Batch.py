
import numpy as np

from ..Helpers import funcs

class Batch(object):
    """docstring for Batch"""
    def __init__(self, data, data_masks, data_targets, domain_targets):
        super(Batch, self).__init__()

        self.data = data
        self.data_masks = data_masks
        self.data_targets = data_targets
        self.domain_targets = domain_targets

    def concatenate(self, oth_batch, training):

        # Arrays to pad
        arrays_to_be_padded = [self.data, oth_batch.data]

        # Paddings
        max_seq_len = max(self.data.shape[1], oth_batch.data.shape[1])

        paddings = [[[0, 0], [0, max_seq_len-self.data.shape[1]]] + [[0, 0]] * (len(self.data.shape)-2),
                    [[0, 0], [0, max_seq_len-oth_batch.data.shape[1]]] + [[0, 0]] * (len(oth_batch.data.shape)-2)]

        # Padding operation
        pad_self_array, pad_oth_batch_array = funcs.pad_nparrays(paddings, arrays_to_be_padded)

        # Arrays to be concatenated
        self_array = [pad_self_array, self.data_masks, self.data_targets, self.domain_targets]
        oth_batch_array = [pad_oth_batch_array, oth_batch.data_masks, oth_batch.data_targets, oth_batch.domain_targets]

        # Concatenate arrays & create new concatenated batch
        concat_arrays = []
        for sarr, barr in zip(self_array, oth_batch_array):
            concat_arrays.append(np.vstack([sarr,barr]))

        # Corrections to concat_arrays
        concat_arrays[1] = np.reshape(concat_arrays[1], [-1])
        if training: concat_arrays[2] = self.data_targets

        return Batch(*concat_arrays)

