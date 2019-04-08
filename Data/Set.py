
import numpy as np

from .Batch import Batch
from .Helpers import funcs

class Set(object):
    def __init__(self, domain_data, batch_size, permute=True):
        super(Set, self).__init__()

        self._current_index = 0

        self._binned_data = domain_data.binned_data
        self._index_to_bin_pos = domain_data.index_to_bin_pos

        self._batch_size = batch_size

        if permute:
            self._permutation = np.random.permutation(len(self._index_to_bin_pos))
        else:
            self._permutation = np.array(range(len(self._index_to_bin_pos)))

        self.data_shape = list(self._get_from_bin(0).data.shape)
        self.data_shape[0] = None
        self.data_shape = [None] + self.data_shape

        self.data_dtype = self._get_from_bin(0).data.dtype
        self.data_ndims = len(self.data_shape)

        self.target_shape = [None, len(self._get_from_bin(0).data_targets)]
        self.target_dtype = self._get_from_bin(0).data_targets.dtype
        self.target_ndims = len(self.target_shape)

        self.domain_shape = [None, len(self._get_from_bin(0).domain_targets)]
        self.domain_dtype = self._get_from_bin(0).domain_targets.dtype
        self.domain_ndims = len(self.domain_shape)

    def _get_from_bin(self, index):
        bin_, pos = self._index_to_bin_pos[index]
        return self._binned_data[bin_][pos]

    def repeat(self):
        self._current_index = 0

    def get_all_data(self):

        # Prepare start & end indexes
        start_idx = 0
        end_idx = len(self._permutation)

        # Retrieving data
        batch = self._get_data(start_idx, end_idx)

        return batch

    def next_batch(self, no_remainder=True):

        # Return none if whole database as been read
        if self._current_index >= len(self._permutation):
            return None

        # Prepare start & end indexes
        start_idx = self._current_index
        end_idx = min(start_idx + self._batch_size, len(self._permutation))

        # Increasing index
        self._current_index = end_idx

        # Let go of remainder
        if end_idx-start_idx != self._batch_size and no_remainder:
            return None

        # Retrieving data
        batch = self._get_data(start_idx, end_idx)

        return batch

    def _get_data(self, start_idx, end_idx):
        # Support arrays setup
        batch_dict = {key: [] for key in ['data', 'data_masks', 'data_targets', 'domain_targets']}

        # Data retrieval
        for i in range(start_idx, end_idx):

            idx = self._permutation[i]
            item = self._get_from_bin(idx)

            batch_dict['data'].append(item.data)
            batch_dict['data_masks'].append(item.data_lengths)
            batch_dict['data_targets'].append(item.data_targets)
            batch_dict['domain_targets'].append(item.domain_targets)

        # Padding sequences to same length
        max_seq_len = max([seq.shape[0] for seq in batch_dict['data']])

        paddings = [[0, max_seq_len-x.shape[0]] for x in batch_dict['data']]
        paddings = [[x] + [[0, 0]] * (len(self.data_shape)-2) for x in paddings] # Adding no pad for feature dimensions in data

        padded_array = funcs.pad_nparrays(paddings, batch_dict['data'])

        # Numpy conversion
        lists = list(batch_dict.values())
        lists[0] = padded_array

        numpy_data = [np.array(arr) for arr in lists]

        return Batch(*numpy_data)

