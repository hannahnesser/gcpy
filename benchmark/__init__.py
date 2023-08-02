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

from .run_benchmark import *
from .modules import *
#from .modules.benchmark_common import *
#from .modules.benchmark_models_vs_obs import *
#from .modules.budget_ox import *
#from .modules.budget_tt import *
#from .modules.meah_oh_from_logs import *
#from .modules.oh_metrics import *
#from .modules.run_1yr_fullchem_benchmark import *
#from .modules.run_1yr_tt_benchmark import *
from .modules.ste_flux import *
