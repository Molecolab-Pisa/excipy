# excipy: machine learning models for a fast estimation of excitonic Hamiltonians
# Copyright (C) 2022 excipy authors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
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

try:
    from . import polar
    from .clib import tmu_cy

    available_polarizable_module = True
except ImportError:
    logging.warning(
        "Something went wrong with the compilation of Cython code."
        " Polarizable module not available."
    )
    available_polarizable_module = False

try:
    from . import intf_ommp

    available_intf_ommp = True
except ImportError:
    logging.warning("pyopenmmpol is not available.")
    available_intf_ommp = False
