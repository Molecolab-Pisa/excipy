import sys
import os
import logging

import numpy as np

from . import database
from . import descriptors
from . import regression
from . import trajectory
from . import util
from . import models

# Cython functions
from . import clib

# We make this import conditional on whether
# the user has compiled the Fortran code.
try:
    from . import polar
    from . import tmu

    available_polarizable_module = True
except ImportError:
    logging.warning(
        "Fortran code tmu.f90 not compiled. This means"
        " you cannot compute the environment polarization"
        " contribution to the Coulomb coupling."
    )
    available_polarizable_module = False

try:
    from . import intf_ommp

    available_intf_ommp = True
except ImportError:
    logging.warning("pyopenmmpol is not available.")
    available_intf_ommp = False
