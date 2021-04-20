"""
ESPEI
"""

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from .logger import _setup_logging
# Makes global logging changes; all new logger instances will be ESPEILogger objects
_setup_logging()

from .citing import ESPEI_BIBTEX, ESPEI_CITATION
__citation__ = ESPEI_CITATION
__bibtex__ = ESPEI_BIBTEX

from espei.paramselect import generate_parameters
from espei.espei_script import run_espei

# swallow warnings during MCMC runs
import warnings
warnings.filterwarnings('ignore', message='Mean of empty slice')
warnings.filterwarnings('ignore', message='invalid value encountered in subtract')
warnings.filterwarnings('ignore', message='invalid value encountered in greater')
warnings.filterwarnings('ignore', message='divide by zero encountered in log')
warnings.filterwarnings('ignore', message='invalid value encountered in true_divide')
warnings.filterwarnings('ignore', message='divide by zero encountered')
warnings.filterwarnings('ignore', message='Ill-conditioned matrix')
warnings.filterwarnings('ignore', message='Singular matrix in solving dual problem')
