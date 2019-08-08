
import tensorflow as tf

from .Layer import Layer

class Graph(object):

    def __init__(self, name, string_spec, input_name, target_name):
        super(Graph, self).__init__()

        self.name = name
        self.input_name = input_name
        self.target_name = target_name

        self.loss = None
        self.hits = None
        self.accuracy = None

        self.layers = {}

        specs = string_spec.split('_')
        for idx, spec in enumerate(specs):
            layer = Layer(self.name, spec, idx)
            self.layers[layer.name] = layer

    def build(self, in_tensor, trg_tensor, init_std):
        curr_tensor = in_tensor
        for i,l in enumerate(self.layers.values()):
            if i == len(self.layers.values())-1 and trg_tensor is not None:
                self.loss, self.hits, self.accuracy = l.build(curr_tensor, init_std, trg_tensor)
            else:
                curr_tensor = l.build(curr_tensor, init_std)
