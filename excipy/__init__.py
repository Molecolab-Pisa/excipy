import sys
import os
import logging

# Disable Tensorflow warnings, infos, messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
import gpflow

gpflow.config.set_default_float(np.float64)

from . import database
from . import descriptors
from . import regression
from . import trajectory
from . import util

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
