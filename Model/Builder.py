
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

    def add_specification(self, spec_str, input_name, target_name):
        graph_spec = Graph(spec_str, input_name, target_name)
        self.graph_specs.append(graph_spec)

    def build_model(self):
        for graph in self.graph_specs:
            if graph.input_name in self.placeholders:
                in_tensor = self.placeholders[graph.input_name]
            else:
                in_tensor = tf.get_default_graph().get_tensor_by_name(graph.input_name + ':0')

            if graph.target_name in self.placeholders:
                trg_tensor = self.placeholders[graph.target_name]
            else:
                trg_tensor = None

            graph.build(in_tensor, trg_tensor, self.init_std)

