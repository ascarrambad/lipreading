
import numpy as np
import tensorflow as tf

from .Helpers import enums

class Trainer(object):

    def __init__(self, epochs, optimizer, accuracy, loss, eval_losses, tensorboard_path=None):
        super(Trainer, self).__init__()

        assert tensorboard_path != ''

        self.epochs = epochs
        self.optimizer = optimizer
        self.optimizer = self.optimizer.minimize(loss)

        self.accuracy = accuracy
        self.eval_losses = eval_losses

        self._tensorboard_path = tensorboard_path
        self.tensorboard_status = False

        self.session = None

    def init_session(self):
        if self.session is not None:
            self.session.close()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)

        if self._tensorboard_path is not None:
            self._setup_tensorboard()

        self.session.run(tf.global_variables_initializer())

    def train(self, train_sets, valid_sets, stopping_type, stopping_patience, feed_builder):
        assert self.session is not None
        assert len(train_sets) > 0

        valid_sets = train_sets + valid_sets

        # Variables init
        self._training_current_best = (0,0)

        # Epochs loop
        summ_idx = 0
        for epoch in range(self.epochs):
            # Load initial batches
            list(map(lambda x: x.repeat(), train_sets))
            batches = list(map(lambda x: x.next_batch(), train_sets))

            # Training
            while None not in batches:
                final_batch = batches[0]
                for batch in batches[1:]:
                    final_batch = final_batch.concatenate(batch, training=True)

                # Graph execution
                train_tensors = [self.optimizer]
                if self.tensorboard_status:
                    train_tensors += [self.summaries]

                feed = feed_builder(epoch, final_batch, True)

                results = self._execute(train_tensors, feed, training=True, step=summ_idx)
                if self.tensorboard_status:
                    self._train_writer.add_summary(results[1], summ_idx)

                # Load new Batches
                batches = list(map(lambda x: x.next_batch(), train_sets))
                summ_idx += 1

            # Testing
            losses_accs = self.test(valid_sets, feed_builder, epoch)

            # Retrieving accuracies for early stopping evaluation
            accs = {st: {dt: vv[-1] for (dt,vv) in v.items()} for (st,v) in losses_accs.items()}

            # Early stopping evaluation
            if self._evaluate_stopping(epoch, accs, stopping_type, stopping_patience):
                best_e, best_v = self._training_current_best

                print('Stopping at [EPOCH {0}] because stop condition has been reached'.format(epoch))
                print('Condition satisfied at [EPOCH {0}], best result: {1:.5f}'.format(best_e, best_v))

                return

    def test(self, test_sets, feed_builder, epoch=None):
        # Make sure batch iterators are reset
        list(map(lambda x: x.repeat(), test_sets))

        # Loss and accuracy support arrays
        losses_accs = {x.type: {} for x in test_sets}

        # For each set
        for tset in test_sets:

            # Current loss and accuracy support arrays
            set_losses = {k: [] for k in range(len(self.eval_losses.values()))}
            set_accs = []

            # Load initial Batch
            batch = tset.next_batch()

            # Tensorboard Summaries
            if self.tensorboard_status and epoch != None and tset.type != enums.SetType.TRAIN:

                feed = feed_builder(epoch, batch, False)
                summ = self._execute([self.summaries], feed, training=False)[0]

                if tset.domain_type == enums.DomainType.SOURCE:
                    self._validS_writer.add_summary(summ, epoch)
                elif tset.domain_type == enums.DomainType.TARGET:
                    self._validT_writer.add_summary(summ, epoch)

            # Testing
            while batch is not None:

                feed = feed_builder(epoch, batch, False)

                # Graph execution
                tensors = list(self.eval_losses.values()) + [self.accuracy]
                res = self._execute(tensors, feed, False)
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
        if epoch is not None:
            print('**** [EPOCH {0}] ****'.format(epoch))
        self._pretty_print(losses_accs)

        return losses_accs

    def _setup_tensorboard(self):
        self._train_writer = tf.summary.FileWriter(self._tensorboard_path + '/train', self.session.graph)
        self._validS_writer = tf.summary.FileWriter(self._tensorboard_path + '/validS', self.session.graph)
        self._validT_writer = tf.summary.FileWriter(self._tensorboard_path + '/validT', self.session.graph)

        self.summaries = tf.summary.merge_all()

        self.tensorboard_status = True

    def _evaluate_stopping(self, epoch, accs, criteria, patience):
        doStop = False
        if criteria != enums.StoppingType.OFF:
            stopValue = accs[criteria.value[0]][criteria.value[1]]

            if stopValue > self._training_current_best[1]:
                self._training_current_best = (epoch, stopValue)
            elif epoch - self._training_current_best[0] > patience:
                doStop = True

        return doStop

    def _execute(self, tensors, feed, training, step=None):

        if training and step is not None:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            res = self.session.run(tensors, feed_dict=feed, options=run_options, run_metadata=run_metadata)
            if self.tensorboard_status:
                self._train_writer.add_run_metadata(run_metadata, 'step-%d' % step)
        else:
            res = self.session.run(tensors, feed)

        return res

    def _pretty_print(self, losses_accs):
        for key in losses_accs.keys():
            print('  [{0}]'.format(key.name))
            for k,v in losses_accs[key].items():
                print('    {0}\n    - Loss('.format(k.name), end='')
                print(' / '.join(self.eval_losses.keys()), end='')
                print('): ', end='')
                str_losses = ['{0:.5f}'.format(x) for x in v[:-1]]
                print(' / '.join(str_losses), end='')
                print('\n    - Accuracy: {0:.5f}\n'.format(v[-1]))


