
import numpy as np

from .Batch import Batch
from .Helpers import funcs

class Set(object):
    def __init__(self, domain_data, batch_size, permute=True): # buffer_size
        super(Set, self).__init__()

        self._current_index = 0

        self._binned_data = domain_data.binned_data
        self._index_to_bin_pos = domain_data.index_to_bin_pos

        self._batch_size = batch_size
        # self._buffer_size = buffer_size

        if permute:
            self._permutation = np.random.permutation(len(self._index_to_bin_pos))
        else:
            self._permutation = np.array(range(len(self._index_to_bin_pos)))

        self.data_shape = list(self._get_from_bin(0).data.shape)
        self.data_shape[0] = None
        self.data_shape = [None] + self.data_shape

        self.data_dtype = self._get_from_bin(0).data.dtype
        self.data_ndims = len(self.data_shape)

        self.target_shape = [None, None]
        self.target_dtype = self._get_from_bin(0).wordtargets.dtype
        self.target_ndims = len(self.target_shape)

        self.domain_shape = [None, None]
        self.domain_dtype = self._get_from_bin(0).speakerlabels.dtype
        self.domain_ndims = len(self.domain_shape)

    def _get_from_bin(self, index):
        bin_, pos = self._index_to_bin_pos[index]
        return self._binned_data[bin_][pos]

    def repeat(self):
        self._current_index = 0

    def next_batch(self):
        # Support arrays setup
        batch_dict = {key: [] for key in ['data', 'data_masks', 'data_targets', 'domain_targets']}

        # Return none if whole database as been read
        if self._current_index >= len(self._permutation):
            return None

        # Extracting data from binned_data
        start_idx = self._current_index
        end_idx = start_idx + self._batch_size

        for i in range(start_idx, end_idx):

            idx = self._permutation[i]
            item = self._get_from_bin(idx)

            batch_dict['data'].append(item.data)
            batch_dict['data_masks'].append(item.wordmask)
            batch_dict['data_targets'].append(item.wordtargets)
            batch_dict['domain_targets'].append(item.speakerlabels)

        # Increasing index
        self._current_index = end_idx

        # Padding sequences to same length
        max_seq_len = max([seq.shape[0] for seq in batch_dict['data']])

        paddings = [[0, max_seq_len-x.shape[0]] for x in batch_dict['data']]
        paddings = [paddings] * len(batch_dict.items()) # Repeting padding for all 4 arrays
        paddings[0] = [[x] + [[0, 0]] * (len(self.data_shape)-2) for x in paddings[0]] # Adding no pad for feature dimensions in data

        padded_arrays = funcs.pad_nparrays(paddings, list(batch_dict.values()))

        numpy_data = [np.array(arr) for arr in padded_arrays]

        return Batch(*numpy_data)

