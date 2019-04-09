
import numpy as np
import tensorflow as tf

class Trainer(object):

    def __init__(self, epochs, learning_rate, graph_specs, placeholders, tensorboard_path=None):
        super(Trainer, self).__init__()

        assert tensorboard_path != ''

        self._graph_specs = graph_specs
        self._placeholders = placeholders
        self.epochs = epochs
        self.learning_rate = learning_rate
        self._tensorboard_path = tensorboard_path

        loss = sum([x.loss for x in self._graph_specs])

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = optimizer.minimize(loss)

        self.session = None

    def init_session(self, session=None):
        if session is None:
            self.session = tf.Session()

    def train(self, train_sets, valid_sets):
        assert self.session is not None
        assert len(train_sets) > 0
        assert len(valid_sets) > 0

        # Variables init and Tensorflow setup
        self.session.run(tf.global_variables_initializer())
        self._setup_tensorboard()

        # Epochs loop
        for epoch in range(self.epochs):
            # Load initial batches
            list(map(lambda x: x.repeat(), train_sets))
            batches = list(map(lambda x: x.next_batch(), train_sets))

            p = float(epoch) / self.epochs
            lambda_ = 2. / (1. + np.exp(-10. * p)) - 1
            # lr = 0.01 / (1. + 10 * p)**0.75

            # Training
            while None not in batches:
                final_batch = batches[0]
                for batch in batches[1:]:
                    final_batch = final_batch.concatenate(batch, training=True)

                # Graph execution
                self._execute([self.optimizer], final_batch, lambda_, training=True)

                # Load new Batches
                batches = list(map(lambda x: x.next_batch(), train_sets))

            # Testing
            print('Epoch: {0}'.format(epoch))
            self.test(valid_sets, new_session=False)

        self.session.close()
        self.session = None

    def test(self, test_sets, new_session=True):

        if new_session:
            self.session.run(tf.global_variables_initializer())
            self._setup_tensorboard()

        batches = list(map(lambda x: x.get_all_data(), test_sets))
        final_batch = batches[0]
        for batch in batches[1:]:
            final_batch = final_batch.concatenate(batch, training=True)

        # Tensors to evaluate
        loss = self._graph_specs[0].loss
        acc = self._graph_specs[0].accuracy

        # Graph execution
        loss_, acc_ = self._execute([loss, acc], final_batch, 0.0, False)
        print('Loss: {0}, Accuracy: {1}'.format(loss_,acc_))

        if new_session:
            self.session.close()
            self.session = None

    def _setup_tensorboard(self):
        if self._tensorboard_path is not None:
            self.train_writer = tf.summary.FileWriter(self._tensorboard_path + '/train', self.session.graph)
            # self.valid_writer = tf.summary.FileWriter(self._tensorboard_path + '/valid', self.session.graph)

            self.summaries = tf.summary.merge_all()

    def _execute(self, tensors, batch, lambda_, training):

        keys = self._placeholders.values()
        values = [batch.data,
                  batch.data_masks,
                  batch.data_targets,
                  batch.domain_targets,
                  lambda_,
                  training]
        feed = dict(zip(keys, values))

        return self.session.run(tensors, feed)


