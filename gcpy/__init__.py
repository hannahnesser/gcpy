"""
GCPY initialization script.  Imports nested packages for convenience.
"""

try:
    from ._version import __version__
except ImportError:
    raise ImportError(
        'gcpy was not properly installed; some functionality '
        'may be not work. If installing from source code, '
        'please re-install in place by running\n'
        '$ pip install -e .'
        '\nElse, please reinstall using your package manager.'
    )

from .append_grid_corners import *
from .constants import *
from .cstools import *
from .date_time import *
from .file_regrid import *
from .grid import *
from .grid_stretching_transforms import *
from .plot import *
from .raveller_1D.py import *
from .regrid import *
from .regrid_restart_file import *
from .units import *
from .util import *
