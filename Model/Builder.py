
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
        self._graph_specs = []
        self._graph_name_idx = {}

    def add_placeholder(self, dtype, shape, name):
        if name not in self.placeholders.keys():
            with tf.variable_scope('Inputs/'):
                plc = tf.placeholder(dtype, shape, name)
                self.placeholders[name] = plc
                return plc
        else:
            raise Exception('Placeholder \'%s\' already exists' % name)

    def add_specification(self, name, spec_str, input_name, target_name):
        if name not in self._graph_name_idx.keys():
            graph_spec = Graph(name, spec_str, input_name, target_name)
            self._graph_specs.append(graph_spec)
            self._graph_name_idx[name] = len(self._graph_specs) - 1
            return graph_spec
        else:
            raise Exception('Graph specification \'%s\' already exists' % name)

    def get_specification(self, name):
        return self._graph_specs[self._graph_name_idx[name]]

    def get_all_specifications(self):
        return self._graph_specs

    def build_model(self, build_order=None):
        if build_order is None:
            build_order = list(range(len(self._graph_specs)))
        else:
            build_order = [self._graph_name_idx[n] for n in build_order]

        for i in build_order:
            graph = self._graph_specs[i]

            if type(graph.input_name) is list:
                in_tensor = list(map(lambda x: self._get_tensor(x), graph.input_name))
            else:
                in_tensor = self._get_tensor(graph.input_name)

            if graph.target_name in self.placeholders:
                trg_tensor = self.placeholders[graph.target_name]
            else:
                trg_tensor = None

            graph.build(in_tensor, trg_tensor, self.init_std)

    def restore_model(self, path):
        tf.reset_default_graph()
        saver = tf.train.import_meta_graph(path + '/graph.meta')

        plc_names = [ op.name for op in tf.get_default_graph().get_operations() if op.type == "Placeholder"]
        self.placeholders = {x.replace('Inputs/',''):self._get_tensor(x) for x in plc_names}

        return saver

    def _get_tensor(self, name):
        if name in self.placeholders:
            return self.placeholders[name]
        else:
            return tf.get_default_graph().get_tensor_by_name(name + ':0')
