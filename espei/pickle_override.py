from pickle import *
from pickle import _Unpickler, whichmodule
import dill
import cloudpickle

dumps = cloudpickle.dumps

_Pickler = cloudpickle.CloudPickler
Pickler = cloudpickle.CloudPickler
