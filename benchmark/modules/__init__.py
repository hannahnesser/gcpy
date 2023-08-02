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

from .benchmark_common import *
from .benchmark_models_vs_obs import *
from .budget_ox import *
from .budget_tt import *
from .meah_oh_from_logs import *
from .oh_metrics import *
from .run_1yr_fullchem_benchmark import *
from .run_1yr_tt_benchmark import *
from .ste_flux import *
