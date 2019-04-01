
class Domain(object):
    def __init__(self, domain, binned_data, index_to_bin_pos):
        super(Domain, self).__init__()

        self.domain = domain
        self.binned_data = binned_data
        self.index_to_bin_pos = index_to_bin_pos
