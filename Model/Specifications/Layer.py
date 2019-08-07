
import re

import tensorflow as tf

from ..Helpers.layers import layer_type

SPECIAL = r'(?P<special>\*?)'
TYPE = r'(?P<type>[A-Z]+)'
NAME = r'(?P<name>\([A-Za-z0-9]+\))?'
LAYERS = r'(?P<sublayers>(([\d]+(t|r|s|i))-?)+)?'
ARGS = r'(!(?P<args>([^-]+-?)+))?'

SPEC = SPECIAL + TYPE + NAME + LAYERS + ARGS

class Layer(object):

    def __init__(self, graph_name, spec_str, index):
        super(Layer, self).__init__()

        self.graph_name = graph_name
        self.index = index

        match = re.match(SPEC, spec_str)

        self.special = match.group('special') is '*'
        self.type = match.group('type')
        self.name = '{0}-{1}'.format(match.group('name').strip('()') if match.group('name') is not None else self.type, index)
        self.sublayers = match.group('sublayers').split('-') if not self.special else []
        self.args = match.group('args').split('-') if match.group('args') is not None else []

        self.extra_params = {}

        if self.type == 'LSTM' or self.type == 'CONVLSTM':
            self.extra_params['SequenceLengthsTensor'] = None
        elif self.type == 'MASKSEQ':
            self.extra_params['MaskIndicesTensor'] = None
        elif self.type == 'DP':
            self.extra_params['TrainingStatusTensor'] = None
        elif self.type == 'GRADFLIP':
            self.extra_params['LambdaTensor'] = None
        elif self.type == 'CUSTOM':
            self.extra_params['CustomFunction'] = lambda x,y: x

        self.tensors = []

    def build(self, in_tensor, init_std, trg_tensor=None):
        with tf.variable_scope(self.graph_name + '-' + self.name):
            if type(in_tensor) is list:
                in_tensor = [tf.identity(t, name='Input-%d'%i) for i,t in enumerate(in_tensor)]
            else:
                in_tensor = tf.identity(in_tensor, name='Input')
            self.tensors.append(in_tensor)

            args = self.args

            if self.extra_params != {}:
                args = [self.extra_params] + args

            if trg_tensor is not None:
                args = args + [trg_tensor]

            if self.special:
                curr_tensor = layer_type[self.type](in_tensor, *args)
            else:
                curr_tensor = in_tensor
                for l in self.sublayers:
                    num_hidden_units = int(l[:-1])
                    activ_func = l[-1]

                    curr_tensor = layer_type[self.type](curr_tensor, num_hidden_units, init_std, activ_func, *args)

            self.tensors.append(curr_tensor)

        return curr_tensor