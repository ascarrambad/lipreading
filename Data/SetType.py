
from .Helpers import consts

class SetType(object):
    def __init__(self, type):
        self.type = type
        if self.type == 'train':
            self.path = consts.TRAIN_PATH
        elif self.type == 'valid':
            self.path = consts.VALID_PATH
        elif self.type == 'test':
            self.path = consts.TEST_PATH
        else:
            raise Exception('Wrong type parameter')
