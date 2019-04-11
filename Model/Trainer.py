
import numpy as np
import tensorflow as tf

from .Helpers import enums

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
        if session is not None:
            self.session.close()

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def train(self, train_sets, valid_sets, stopping_type, stopping_patience):
        assert self.session is not None
        assert len(train_sets) > 0

        valid_sets = train_sets + valid_sets

        # Variables init and Tensorflow setup
        self._training_current_best = (0,0)
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
            losses, _ = self.test(valid_sets)

            keys = list(map(lambda x: (x.type, x.domain_type), valid_sets))
            if self._evaluate_stopping(epoch, losses, stopping_type, stopping_patience):
                print('Stopping at epoch %d because stop condition has been reached' % epoch)
                return

    def test(self, test_sets):
        # Make sure batch iterators are reset
        list(map(lambda x: x.repeat(), test_sets))

        # Loss and accuracy support arrays
        losses = []
        accs = []

        # For each set
        for tset in test_sets:

            # Current loss and accuracy support arrays
            set_loss = []
            set_acc = []

            # Load initial Batch
            batch = tset.next_batch()

            # Testing
            while batch is not None:
                # Tensors to evaluate
                loss = self._graph_specs[0].loss
                acc = self._graph_specs[0].accuracy

                # Graph execution
                l, a = self._execute([loss, acc], batch, 0.0, False)
                set_loss.append(l)
                set_acc.append(a)

                # Load new Batch
                batch = tset.next_batch()

            # Compute mean and print
            set_loss = np.mean(np.array(set_loss))
            set_acc = np.mean(np.array(set_acc))
            losses.append(set_loss)
            accs.append(set_acc)
            print('Set[{0}-{1}] Loss: {2}, Accuracy: {3}'.format(tset.type.name, tset.domain_type.name,set_loss,set_acc))

        return losses, accs

    def _setup_tensorboard(self):
        if self._tensorboard_path is not None:
            self.train_writer = tf.summary.FileWriter(self._tensorboard_path + '/train', self.session.graph)
            self.validS_writer = tf.summary.FileWriter(self._tensorboard_path + '/validS', self.session.graph)
            self.validT_writer = tf.summary.FileWriter(self._tensorboard_path + '/validT', self.session.graph)

            self.summaries = tf.summary.merge_all()

    def _evaluate_stopping(self, epoch, losses, criteria, patience):
        doStop = False
        if criteria != enums.StoppingType.OFF:
            stopValue = losses[criteria.value]

            if stopValue > self._training_current_best[1]:
                self._training_current_best = (epoch,stopValue)
            elif epoch - self._training_current_best[0] > patience:
                doStop = True

        return doStop

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


