
import numpy as np
import tensorflow as tf

from .Helpers import enums

class AdvTrainer(object):

    def __init__(self, epochs, learning_rate, graph_specs, placeholders, tensorboard_path=None):
        super(AdvTrainer, self).__init__()

        assert tensorboard_path != ''

        self._graph_specs = graph_specs
        self._placeholders = placeholders
        self.epochs = epochs
        self.learning_rate = learning_rate
        self._tensorboard_path = tensorboard_path

        self.losses = [x.loss for x in self._graph_specs if x.loss != None]
        jloss = tf.identity(sum(self.losses), name='JointLoss')
        self.losses.append(jloss)

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = optimizer.minimize(jloss)

        self.session = None

    def init_session(self, session=None):
        if session is not None:
            self.session.close()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)
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
            print('EPOCH [{0}]'.format(epoch))
            losses_accs = self.test(valid_sets)

            accs = {st: {dt: vv[1] for (dt,vv) in v.items()} for (st,v) in losses_accs.items()}
            if self._evaluate_stopping(epoch, accs, stopping_type, stopping_patience):
                print('Stopping at EPOCH [{0}] because stop condition has been reached'.format(epoch))
                return

    def test(self, test_sets):
        # Make sure batch iterators are reset
        list(map(lambda x: x.repeat(), test_sets))

        # Loss and accuracy support arrays
        losses_accs = {x.type: {} for x in test_sets}

        # For each set
        for tset in test_sets:

            # Current loss and accuracy support arrays
            set_losses = {k: [] for k in range(len(self.losses))}
            set_accs = []

            # Load initial Batch
            batch = tset.next_batch()

            # Testing
            while batch is not None:
                # Accuracy tensor
                acc = self._graph_specs[0].accuracy

                # Graph execution
                res = self._execute(self.losses + [acc], batch, 0.0, False)
                for i,v in enumerate(res[:-1]):
                    set_losses[i].append(v)
                set_accs.append(res[-1])

                # Load new Batch
                batch = tset.next_batch()

            # Compute mean
            set_losses = [np.mean(np.array(x)) for x in set_losses.values()]
            set_accs = np.mean(np.array(set_accs))
            losses_accs[tset.type][tset.domain_type] = (*set_losses, set_accs)

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

    def _pretty_print(self, losses_accs):
        for key in losses_accs.keys():
            print('* ' + key.name)
            for k,v in losses_accs[key].items():
                print('  [{0}]\n  Loss(Seq / Adv / Joint): {1:.4f} / {2:.4f} / {3:.4f}\n  Accuracy: {4:.4f}\n'.format(k.name, *v))


