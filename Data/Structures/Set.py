
import numpy as np

from .Batch import Batch
from ..Helpers import funcs

class Set(object):
    def __init__(self, domain_data, batch_size, truncate_remainder=False, permute=True):
        super(Set, self).__init__()

        self.type = domain_data.set_type
        self.domain_type = domain_data.type

        self._current_index = 0

        self._binned_data = domain_data.binned_data
        self._index_to_bin_pos = domain_data.index_to_bin_pos

        self._batch_size = batch_size
        self._truncate_remainder = truncate_remainder

        if permute:
            self._permutation = np.random.permutation(len(self._index_to_bin_pos))
        else:
            self._permutation = np.array(range(len(self._index_to_bin_pos)))

        self.data_shape = list(self._get_from_bin(0).data.shape)
        self.data_shape[0] = None
        self.data_shape = [None] + self.data_shape

        self.data_dtype = self._get_from_bin(0).data.dtype
        self.data_ndims = len(self.data_shape)

        self.target_shape = [None, len(self._get_from_bin(0).data_target)]
        self.target_dtype = self._get_from_bin(0).data_target.dtype
        self.target_ndims = len(self.target_shape)

        self.domain_shape = [None, len(self._get_from_bin(0).domain_target)]
        self.domain_dtype = self._get_from_bin(0).domain_target.dtype
        self.domain_ndims = len(self.domain_shape)

    @property
    def count(self):
        return len(self._permutation)

    def repeat(self, permute=True):
        self._current_index = 0

        if permute:
            self._permutation = np.random.permutation(len(self._index_to_bin_pos))

    @property
    def all(self):

        # Prepare start & end indexes
        start_idx = 0
        end_idx = len(self._permutation)

        # Retrieving data
        batch = self._get_batch(start_idx, end_idx)

        return batch

    @property
    def next_batch(self):

        # Return none if whole database as been read
        if self._current_index >= len(self._permutation):
            return None

        # Prepare start & end indexes
        start_idx = self._current_index
        end_idx = min(start_idx + self._batch_size, len(self._permutation))

        # Increasing index
        self._current_index = end_idx

        # Optionally truncate remainder
        if self._truncate_remainder and end_idx-start_idx != self._batch_size:
            return None

        # Retrieve data
        batch = self._get_batch(start_idx, end_idx)

        return batch

    def _get_batch(self, start_idx, end_idx):
        # Support arrays setup
        batch_dict = {key: [] for key in ['data', 'data_opt', 'data_lengths', 'data_targets', 'domain_targets']}

        # Data retrieval
        for i in range(start_idx, end_idx):

            idx = self._permutation[i]
            item = self._get_from_bin(idx)

            batch_dict['data'].append(item.data)
            batch_dict['data_opt'].append(item.data_opt)
            batch_dict['data_lengths'].append(item.data_length)
            batch_dict['data_targets'].append(item.data_target)
            batch_dict['domain_targets'].append(item.domain_target)

        # Padding sequences to same length
        max_seq_len = max([seq.shape[0] for seq in batch_dict['data']])

        paddings = [[0, max_seq_len-x.shape[0]] for x in batch_dict['data']]
        paddings = [[x] + [[0, 0]] * (len(self.data_shape)-2) for x in paddings] # Adding no pad for feature dimensions in data

        padded_array = funcs.pad_nparrays(paddings, batch_dict['data'])

        # OPTIONALLY ENABLED
        if not any(x is None for x in batch_dict['data_opt']):
            max_seq_len = max([seq.shape[0] for seq in batch_dict['data_opt']])

            paddings = [[0, max_seq_len-x.shape[0]] for x in batch_dict['data_opt']]
            paddings = [[x] + [[0, 0]] * (len(self.data_shape)-2) for x in paddings] # Adding no pad for feature dimensions in data

            padded_array2 = funcs.pad_nparrays(paddings, batch_dict['data_opt'])

        # Numpy conversion
        lists = list(batch_dict.values())
        lists[0] = padded_array

        # OPTIONALLY ENABLED
        if not any(x is None for x in batch_dict['data_opt']):
            lists[1] = padded_array2

        numpy_data = [np.array(arr) for arr in lists]

        return Batch(*numpy_data)

    def _get_from_bin(self, index):
        bin_, pos = self._index_to_bin_pos[index]
        return self._binned_data[bin_][pos]
