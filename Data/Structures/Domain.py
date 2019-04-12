
class Domain(object):
    def __init__(self, type_, set_type, binned_data, index_to_bin_pos):
        super(Domain, self).__init__()

        self.type = type_
        self.set_type = set_type
        self.binned_data = binned_data
        self.index_to_bin_pos = index_to_bin_pos
