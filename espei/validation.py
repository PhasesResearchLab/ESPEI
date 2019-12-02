
__all__ = ['schema']

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
    schema = ESPEIValidator(yaml.load(f, Loader=yaml.FullLoader))
