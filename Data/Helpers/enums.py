
from enum import Enum

from . import consts

class SetType(Enum):

    TRAIN, VALID, TEST = range(3)

    @property
    def path(self):
        if self.name == 'TRAIN':
            return consts.TRAIN_PATH
        elif self.name == 'VALID':
            return consts.VALID_PATH
        elif self.name == 'TEST':
            return consts.TEST_PATH

class DomainType(Enum):
    SOURCE, TARGET, ALL = range(3)