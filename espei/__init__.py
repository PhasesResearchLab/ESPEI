"""
ESPEI
"""

# Set the version of espei
try:
    from ._dev import get_version
    # We have a local (editable) installation and can get the version based on the
    # source control management system at the project root.
    __version__ = get_version(root='..', relative_to=__file__)
    del get_version
except ImportError:
    # Fall back on the metadata of the installed package
    try:
        from importlib.metadata import version
    except ImportError:
        # backport for Python<3.8
        from importlib_metadata import version
    __version__ = version("espei")
    del version


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
