
import re

import tensorflow as tf

from ..Helpers.Layers import layer_type

SPECIAL = r'(?P<special>\*?)'
TYPE = r'(?P<type>[A-Z]+)'
NAME = r'(?P<name>\([A-Za-z0-9]+\))?'
LAYERS = r'(?P<sublayers>(([\d]+(t|r|s|i))-?)+)?'
ARGS = r'(!(?P<args>([^-]+-?)+))?'

SPEC = SPECIAL + TYPE + NAME + LAYERS + ARGS

class Layer(object):

    def __init__(self, spec_str, index):
        super(Layer, self).__init__()

        self.index = index

        match = re.match(SPEC, spec_str)

        self.special = match.group('special') is '*'
        self.type = match.group('type')
        self.name = match.group('name').strip('()') if match.group('name') is not None else '{0}-{1}'.format(self.type, index)
        self.sublayers = match.group('sublayers').split('-') if not self.special else []
        self.args = match.group('args').split('-') if match.group('args') is not None else []

        self.tensors = []

    def build(self, in_tensor, init_std):
        with tf.name_scope(self.name):
            in_tensor = tf.identity(in_tensor, name='Input')
            self.tensors.append(in_tensor)
            if self.special:
                out_tensor = layer_type[self.type](in_tensor, *self.args)
            else:
                curr_tensor = in_tensor
                for l in self.sublayers:
                    num_hidden_units = int(l[:-1])
                    activ_func = l[-1]

                    out_tensor = layer_type[self.type](curr_tensor, num_hidden_units, init_std, activ_func, *self.args)
                    self.tensors.append(out_tensor)

        curr_tensor = out_tensor

        return curr_tensor