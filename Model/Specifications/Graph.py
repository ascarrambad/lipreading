
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
        for l in self.layers.values():
            curr_tensor = l.build(curr_tensor, init_std)

        if trg_tensor is not None:
            with tf.variable_scope(self.name + '-CLASSIF'):
                out_tensor = tf.identity(curr_tensor, name='Logits')

                cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=trg_tensor,
                                                                           logits=out_tensor,
                                                                           name='SoftmaxCrossEntropy')
                self.loss = tf.reduce_mean(cross_entropy, name='Loss')

                self.hits = tf.equal(tf.argmax(out_tensor, axis=1), tf.argmax(trg_tensor, axis=1), name='Hits')
                self.accuracy = tf.reduce_mean(tf.cast(self.hits, tf.float32), name='Accuracy')

                tf.summary.scalar('Loss', self.loss)
                tf.summary.scalar('Accuracy', self.accuracy)