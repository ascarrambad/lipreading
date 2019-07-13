
import subprocess
import time
import os
import itertools
import random
import re

import numpy as np

os.environ['OMP_NUM_THREADS'] = '1'

class BatchExperiment(object):
    def __init__(self, script_names, gpus, exp_params, db_path):

        self._load_cur = ['/','-','\\','|']

        self._script_names = script_names
        self._gpus = set(gpus)
        self._db_path = db_path

        # Generate all possible combinations of parameters
        exp_params_keys = sorted(exp_params.keys())

        exp_params_values = []
        for key in exp_params_keys:
            if type(exp_params[key]) is list:
                exp_params_values.append(exp_params[key])
            else:
                exp_params_values.append([exp_params[key]])
        combs = itertools.product(*exp_params_values)

        self.exp_params = []
        for c in combs:
            self.exp_params.append(dict(zip(exp_params_keys, c)))

        # iteration, RunningTasks is indexed by GPU
        self._running_tasks = dict()

    def run(self):
        for script in self._script_names:
            for params in self.exp_params:
                self._slot_request()

                # now start
                nextGPU = min(self._gpus - set(self._running_tasks.keys()))

                # this is a bit more complicated
                par_list = ["DBPath='%s'" % self._db_path]
                for (key, val) in params.items():
                    # make parameter string
                    if isinstance(val, float) or isinstance(val, int):
                        p_str = '%s=%s' % (key, val)
                    elif isinstance(val, str):
                        p_str = "%s='%s'" % (key, val)
                    elif val is None:
                        p_str = "%s=None" % key
                    else:
                        raise Exception('AAARGH')
                    par_list.append(p_str)

                arg_seq = ['python', script+'.py', 'with'] + par_list

                print('Now start: %s' % str(arg_seq))

                environment = os.environ.copy()
                environment['CUDA_VISIBLE_DEVICES'] = str(nextGPU)

                P = subprocess.Popen(arg_seq, env=environment)

                self._running_tasks[nextGPU] = P

        while len(self._running_tasks) != 0:
            print('Waiting for all experiments to end ... %s' % next(self._load_cur), end='\r')
            self._remove_task()

    def _slot_request(self):
        # wait for free slot
        while len(self._running_tasks) >= len(self._gpus):
            print('Waiting for slot ... %s' % next(self._load_cur), end='\r')
            self._remove_task()

    def _remove_task(self):
        for gpu,task in self._running_tasks.items():
            if task.poll() is not None:
                # remove finished task
                print('Removing task %s' % str(task))
                del self._running_tasks[gpu]
                break
        else:
            time.sleep(2)