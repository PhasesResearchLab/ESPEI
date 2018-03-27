"""
ESPEI
"""

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

# the hackiest hack such that importing pickle will import our model that overrides the dump function
# subclasses made later will probably be maintained.
# we are overrideing a builtin, which is HORRIBLE
# from espei import pickle_override
# import sys
# # we want to be able to get back the old pickle for dask/distributed
# sys.modules['pickle'] = sys.modules['espei.pickle_override']

import os
import yaml
from cerberus import Validator

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

# extension for iseven
class ESPEIValidator(Validator):
    def _validate_iseven(self, iseven, field, value):
        """ Test the oddity of a value.

        The rule's arguments are validated against this schema:
        {'type': 'boolean'}
        """
        if iseven and bool(value & 1):
            self._error(field, "Must be an even number")

with open(os.path.join(MODULE_DIR, 'input-schema.yaml')) as f:
    schema = ESPEIValidator(yaml.load(f))

from espei.paramselect import generate_parameters
from espei.mcmc import mcmc_fit
from espei.espei_script import run_espei

# swallow warnings during MCMC runs
import warnings
warnings.filterwarnings('ignore', message='Mean of empty slice')
warnings.filterwarnings('ignore', message='invalid value encountered in subtract')
warnings.filterwarnings('ignore', message='invalid value encountered in greater')
warnings.filterwarnings('ignore', message='divide by zero encountered in log')
warnings.filterwarnings('ignore', message='invalid value encountered in true_divide')
warnings.filterwarnings('ignore', message='divide by zero encountered')
