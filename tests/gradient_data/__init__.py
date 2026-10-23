"""
Invented A-B-C-D database and datasets for the parameter gradient tests, see generate.py.
"""
import json
import os

GRADIENT_DATA_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(GRADIENT_DATA_DIR, 'abcd.tdb')) as fp:
    ABCD_GRADIENT_TDB = fp.read()

# Parameters for the central-difference checks; together they touch every phase in each system's data
FD_SYMBOLS = {
    'cu_mg': ['VV0000', 'VV0003', 'VV0007', 'VV0008', 'VV0009', 'VV0015', 'VV0016', 'VV0017'],
    'ternary': ['VV0000', 'VV0008', 'VV0014', 'VV0016', 'VV0022', 'VV0026', 'VV0028', 'VV0030'],
    'quaternary': ['VV0000', 'VV0012', 'VV0016', 'VV0022', 'VV0024', 'VV0027', 'VV0029', 'VV0030'],
}


def load_gradient_datasets(system):
    """Generated datasets of the ternary or quaternary system, by name."""
    with open(os.path.join(GRADIENT_DATA_DIR, f'{system}.json')) as fp:
        return json.load(fp)
