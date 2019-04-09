
import tensorflow as tf

from .Layer import Layer

class Graph(object):

    def __init__(self, string_spec, input_name, target_name):
        super(Graph, self).__init__()

        self.input_name = input_name
        self.target_name = target_name

        specs = string_spec.split('_')

        self.layers = {}
        for idx, spec in enumerate(specs):
            layer = Layer(spec, idx)
            self.layers[layer.name] = layer

    def build(self, in_tensor, trg_tensor, init_std):
        curr_tensor = in_tensor
        for l in self.layers.values():
            curr_tensor = l.build(curr_tensor, init_std)

        if trg_tensor is not None:
            out_tensor = curr_tensor

            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=trg_tensor,
                                                                       logits=out_tensor)
            self.loss = tf.reduce_mean(cross_entropy)

            self.hits = tf.equal(tf.argmax(out_tensor, axis=1), tf.argmax(trg_tensor, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(self.hits, tf.float32))