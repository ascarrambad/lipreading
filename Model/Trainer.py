
import numpy as np
import tensorflow as tf

from .Helpers import enums

class Trainer(object):

    def __init__(self, epochs, optimizer, accuracy, loss, eval_losses, tensorboard_path=None, model_path=None):
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
        self._training_current_best = None

        self._model_path = model_path
        self.model_saver_status = model_path is not None

        if self.model_saver_status:
            self.saver = tf.train.Saver(max_to_keep=1)
            self.saver.export_meta_graph(model_path + '/graph.meta')

    def init_session(self):
        if self.session is not None:
            self.session.close()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)

        if self._tensorboard_path is not None:
            self._setup_tensorboard()

        self.session.run(tf.global_variables_initializer())

    def train(self, train_sets, valid_sets, batched_valid, stopping_type, stopping_value, stopping_patience, feed_builder):
        assert self.session is not None
        assert len(train_sets) > 0

        valid_sets = train_sets + valid_sets

        # Variables init
        self._training_current_best = (-1,-9999)
        tensorb_index = 0
        self._last_tensorb_index = [0]*len(valid_sets)

        # Epochs loop
        for epoch in range(self.epochs):
            # Load initial batches
            list(map(lambda x: x.repeat(), train_sets))
            batches = list(map(lambda x: x.next_batch, train_sets))

            # Training
            while None not in batches:
                final_batch = batches[0]
                for batch in batches[1:]:
                    final_batch = final_batch.concatenate(batch, training=True)

                # Graph execution
                train_tensors = [self.optimizer]
                if self.tensorboard_status:
                    train_tensors += [self.summaries]

                feeds = feed_builder(epoch, final_batch, True)
                if not hasattr(feeds, '__iter__'): feeds = [feeds]
                for f in feeds:
                    results = self.session.run(train_tensors, f)
                    if self.tensorboard_status:
                        self._tboard_writers[enums.SetType.TRAIN][enums.DomainType.SOURCE].add_summary(results[1], tensorb_index)
                    tensorb_index += 1

                # Load new Batches
                batches = list(map(lambda x: x.next_batch, train_sets))

            # Testing
            losses_accs = self._test(valid_sets, batched_valid, feed_builder, epoch)

            if stopping_type is not enums.StoppingType.OFF:
                # Retrieving losses/accuracies for early stopping evaluation
                accs = {st: {dt: stopping_value.value[0]*vv[stopping_value.value[1]] for (dt,vv) in v.items()} for (st,v) in losses_accs.items()}

                # Early stopping evaluation
                if self._evaluate_stopping(epoch, accs, stopping_type, stopping_patience):
                    best_e, best_v = self._training_current_best

                    print('Stopping at [EPOCH {0}] because stop condition has been reached'.format(epoch))
                    print('Condition satisfied at [EPOCH {0}], best result: {1:.5f}'.format(best_e, best_v))

                    return

    def test(self, test_sets, feed_builder, batched=True, restore_best=True):
        assert self.session is not None
        assert len(test_sets) > 0

        if restore_best and self.model_saver_status:
            self.saver = tf.train.import_meta_graph(self._model_path + '/graph.meta')
            path = tf.train.latest_checkpoint(self._model_path)
            self.saver.restore(self.session, path)

        self._test(test_sets, batched, feed_builder)

    def _test(self, test_sets, batched, feed_builder, epoch=None):
        # Make sure batch iterators are reset
        list(map(lambda x: x.repeat(), test_sets))

        # Loss and accuracy support arrays
        losses_accs = {x.type: {} for x in test_sets}

        # For each set
        for i,tset in enumerate(test_sets):

            # Current loss and accuracy support arrays
            set_losses = {k: [] for k in range(len(self.eval_losses.values()))}
            set_accs = []

            # Load initial Batch
            batch = tset.next_batch if batched else tset.all

            # Tensors to evaluate
            tensors = list(self.eval_losses.values()) + [self.accuracy]

            # Tensorboard Summaries
            if self.tensorboard_status and epoch != None and not (tset.type == enums.SetType.TRAIN and tset.domain_type == enums.DomainType.SOURCE):
                tensors.append(self.summaries)

            # Graph execution
            while batch is not None:
                feeds = feed_builder(epoch if epoch != None else 0, batch, False)
                if not hasattr(feeds, '__iter__'): feeds = [feeds]
                for f in feeds:
                    res = self.session.run(tensors, f)

                    acc_idx = -1
                    if self.tensorboard_status and epoch != None and not (tset.type == enums.SetType.TRAIN and tset.domain_type == enums.DomainType.SOURCE):
                        self._tboard_writers[tset.type][tset.domain_type].add_summary(res[-1], self._last_tensorb_index[i])
                        self._last_tensorb_index[i] += 1
                        acc_idx = -2

                    for i,v in enumerate(res[:acc_idx]):
                        set_losses[i].append(v)
                    set_accs.append(res[acc_idx])

                # Load new Batch
                batch = tset.next_batch if batched else None

            # Compute mean
            set_losses = [np.mean(np.array(x)) for x in set_losses.values()]
            set_accs = np.mean(np.array(set_accs))

            # Save Results
            losses_accs[tset.type][tset.domain_type] = (*set_losses, set_accs)

        # Printing results
        if epoch is not None:
            print('**** [EPOCH {0}] ****'.format(epoch))
        self._pretty_print(losses_accs)

        return losses_accs

    def _setup_tensorboard(self):
        self._tboard_writers = {x: {y: tf.summary.FileWriter(self._tensorboard_path +'/'+ x.name+'-'+y.name, self.session.graph)
                                    for y in enums.DomainType}
                                for x in enums.SetType}

        self.summaries = tf.summary.merge_all()

        self.tensorboard_status = True

    def _evaluate_stopping(self, epoch, accs, criteria, patience):
        assert criteria is not enums.StoppingType.OFF

        doStop = False
        stopValue = accs[criteria.value[0]][criteria.value[1]]

        if stopValue > self._training_current_best[1]:
            self._training_current_best = (epoch, stopValue)

            if self.model_saver_status:
                self.saver.save(sess=self.session,
                                save_path=self._model_path + '/model',
                                global_step=epoch,
                                write_meta_graph=False)

        elif epoch - self._training_current_best[0] > patience:
            doStop = True

        return doStop

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


