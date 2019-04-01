
import numpy as np

from .Batch import Batch
from .Helpers import funcs

class Set(object):
    def __init__(self, domain_data, batch_size, permute=True): # buffer_size
        super(Set, self).__init__()

        self.__current_index = 0

        self.__binned_data = domain_data.binned_data
        self.__index_to_bin_pos = domain_data.index_to_bin_pos

        self.__batch_size = batch_size
        # self.__buffer_size = buffer_size

        if permute:
            self.__permutation = np.random.permutation(len(self.__index_to_bin_pos))
        else:
            self.__permutation = np.array(range(len(self.__index_to_bin_pos)))

    def __get_from_bin(self, index):
        bbin, pos = self.__index_to_bin_pos[index]
        return self.__binned_data[bbin][pos]

    def repeat(self):
        self.__current_index = 0

    def next_batch(self):
        # Support arrays setup
        batch_dict = {key: [] for key in ['data', 'data_masks', 'data_targets', 'domain_targets']}

        # Return none if whole database as been read
        if self.__current_index >= len(self.__permutation):
            return None

        # Extracting data from binned_data
        start_idx = self.__current_index
        end_idx = start_idx + self.__batch_size

        for i in range(start_idx, end_idx):

            idx = self.__permutation[i]
            item = self.__get_from_bin(idx)

            batch_dict['data'].append(item.data)
            batch_dict['data_masks'].append(item.wordmask)
            batch_dict['data_targets'].append(item.wordtargets)
            batch_dict['domain_targets'].append(item.speakerlabels)

        # Increasing index
        self.__current_index = end_idx + 1

        # Padding sequences to same length
        max_seq_len = max([seq.shape[0] for seq in batch_dict['data']])

        paddings = [[0, max_seq_len-x.shape[0]] for x in batch_dict['data']]
        paddings = [paddings] * len(batch_dict.values()) # Repeting padding for all 4 arrays
        paddings[0] = [[x] + [[0, 0]] * 2 for x in paddings[0]] # Adding no pad for feature dimensions in data

        padded_arrays = funcs.pad_nparrays(paddings, list(batch_dict.values()))

        numpy_data = [np.array(arr) for arr in padded_arrays]

        return Batch(*numpy_data)


