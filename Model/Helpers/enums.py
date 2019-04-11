
from enum import Enum

class StoppingType(Enum):
    OFF, SOURCETRAIN, TARGETTRAIN, SOURCEVALID, TARGETVALID = range(5)