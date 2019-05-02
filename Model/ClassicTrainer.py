
import numpy as np
import tensorflow as tf

from .Helpers import enums

class ClassicTrainer(object):

    def __init__(self, epochs, learning_rate, graph_specs, placeholders, tensorboard_path=None):
        super(ClassicTrainer, self).__init__()

        assert tensorboard_path != ''

        self._graph_specs = graph_specs
        self._placeholders = placeholders
        self.epochs = epochs
        self.learning_rate = learning_rate
        self._tensorboard_path = tensorboard_path

        self.loss = self._graph_specs[0].loss

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = optimizer.minimize(self.loss)

        self.session = None

    def init_session(self, session=None):
        if session is not None:
            self.session.close()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)
        self.session.run(tf.global_variables_initializer())

    def train(self, train_set, valid_sets, stopping_type, stopping_patience):
        assert self.session is not None

        valid_sets = [train_set] + valid_sets

        # Variables init and Tensorflow setup
        self._training_current_best = (0,0)
        self._setup_tensorboard()

        # Epochs loop
        for epoch in range(self.epochs):
            # Load initial batches
            train_set.repeat()
            batch = train_set.next_batch()

            # Training
            while batch is not None:
                # Graph execution
                self._execute([self.optimizer], batch)

                # Load new Batches
                batch = train_set.next_batch()

            # Testing
            print('EPOCH [{0}]'.format(epoch))
            losses_accs = self.test(valid_sets)

            # Retrieving accuracies for early stopping evaluation
            accs = {st: {dt: vv[1] for (dt,vv) in v.items()} for (st,v) in losses_accs.items()}

            # Early stopping evaluation
            if self._evaluate_stopping(epoch, accs, stopping_type, stopping_patience):
                best_e, best_v = self._training_current_best

                print('Stopping at EPOCH [{0}] because stop condition has been reached'.format(epoch))
                print('Condition satisfied at EPOCH [{0}], best result: {1}'.format(best_e, best_v))

                return

    def test(self, test_sets):
        # Make sure batch iterators are reset
        list(map(lambda x: x.repeat(), test_sets))

        # Loss and accuracy support arrays
        losses_accs = {x.type: {} for x in test_sets}

        # For each set
        for tset in test_sets:

            # Current loss and accuracy support arrays
            set_losses = []
            set_accs = []

            # Load initial Batch
            batch = tset.next_batch()

            # Testing
            while batch is not None:
                # Accuracy tensor
                acc = self._graph_specs[0].accuracy

                # Graph execution
                loss_, acc_ = self._execute([self.loss, acc], batch)
                set_losses.append(loss_)
                set_accs.append(acc_)

                # Load new Batch
                batch = tset.next_batch()

            # Compute mean
            set_losses = np.mean(np.array(set_losses))
            set_accs = np.mean(np.array(set_accs))
            losses_accs[tset.type][tset.domain_type] = (set_losses, set_accs)

        # Printing results
        self._pretty_print(losses_accs)

        return losses_accs

    def _setup_tensorboard(self):
        if self._tensorboard_path is not None:
            self.train_writer = tf.summary.FileWriter(self._tensorboard_path + '/train', self.session.graph)
            self.validS_writer = tf.summary.FileWriter(self._tensorboard_path + '/validS', self.session.graph)
            self.validT_writer = tf.summary.FileWriter(self._tensorboard_path + '/validT', self.session.graph)

            self.summaries = tf.summary.merge_all()

    def _evaluate_stopping(self, epoch, accs, criteria, patience):
        doStop = False
        if criteria != enums.StoppingType.OFF:
            stopValue = accs[criteria.value[0]][criteria.value[1]]

            if stopValue > self._training_current_best[1]:
                self._training_current_best = (epoch,stopValue)
            elif epoch - self._training_current_best[0] > patience:
                doStop = True

        return doStop

    def _execute(self, tensors, batch):

        keys = self._placeholders.values()
        values = [batch.data,
                  batch.data_masks,
                  batch.data_targets]
        feed = dict(zip(keys, values))

        return self.session.run(tensors, feed)

    def _pretty_print(self, losses_accs):
        for key in losses_accs.keys():
            print('* ' + key.name)
            for k,v in losses_accs[key].items():
                print('  [{0}]\n  Loss: {1:.4f}\n  Accuracy: {2:.4f}\n'.format(k.name, *v))
