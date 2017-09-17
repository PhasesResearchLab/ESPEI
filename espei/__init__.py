from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

import os
import yaml
from cerberus import Validator

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(MODULE_DIR, 'input-schema.yaml')) as f:
    schema = Validator(yaml.load(f))

from espei.paramselect import fit
