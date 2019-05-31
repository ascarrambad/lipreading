
from enum import Enum
from Data import SetType, DomainType

class StoppingType(Enum):
    OFF = (None, None)
    SOURCETRAIN = (SetType.TRAIN, DomainType.SOURCE)
    TARGETTRAIN = (SetType.TRAIN, DomainType.TARGET)
    SOURCEVALID = (SetType.VALID, DomainType.SOURCE)
    TARGETVALID = (SetType.VALID, DomainType.TARGET)

class StoppingValue(Enum):
    LOSS = (-1,0)
    ACCURACY = (1,-1)