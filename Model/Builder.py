
import numpy as np
import tensorflow as tf

from .Specifications.Graph import Graph

# Each layer type must be separated by '_', the type is at the beginning in all caps
# Each layer type can have a name enclosed by brackets '(FooBar42)'
# Each layer type can be constituted by multiple layers, each of *n* hidden units separated by '-'
# Each layer must have an activation function indicated by 't'anh, 'r'elu, 's'igmoid, 'i'dentity
# Each layer type can have arguments at the end separated by '-', must be separated by layers by '!'

# N.B.
# There are special layers types that can accept only arguments as parameters. They start with '*'.

# e.g.
# '*FLATFEAT(pippo)_FC(pluto)128t-64l_*DP(paperino)!0.5_*CLASSIF(out)'

class Builder(object):

    def __init__(self, init_std):
        super(Builder, self).__init__()

        self.init_std = init_std

        self.placeholders = {}
        self.graph_specs = []

    def add_placeholder(self, dtype, shape, name):
        with tf.name_scope('Inputs/'):
            plc = tf.placeholder(dtype, shape, name)
            self.placeholders[name] = plc

    def add_main_specification(self, spec_str, input_name, target_name):
        graph_spec = Graph(spec_str, input_name, target_name)
        self.graph_specs.insert(0, graph_spec)

    def add_specification(self, spec_str, input_name, target_name):
        graph_spec = Graph(spec_str, input_name, target_name)
        self.graph_specs.append(graph_spec)

    def build_model(self, build_order=None):
        if build_order is None:
            build_order = list(range(len(self.graph_specs)))

        for i in build_order:
            graph = self.graph_specs[i]

            if type(graph.input_name) is list:
                in_tensor = list(map(lambda x: self._get_tensor(x), graph.input_name))
            else:
                in_tensor = self._get_tensor(graph.input_name)

            if graph.target_name in self.placeholders:
                trg_tensor = self.placeholders[graph.target_name]
            else:
                trg_tensor = None

            graph.build(in_tensor, trg_tensor, self.init_std)

    def _get_tensor(self, name):
        if name in self.placeholders:
            return self.placeholders[name]
        else:
            return tf.get_default_graph().get_tensor_by_name(name + ':0')
