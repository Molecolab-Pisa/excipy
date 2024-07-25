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
"""
Implementation of the available models

Each model is a collection of functions that takes as input a molecule.

Each function uses properties of the molecule object,
with no checks on whether these properties are there.

If you add more models, use the same pattern, and if you want to use a
property that the Molecule object does not posses, implement that property.

If all models are coherent, then in the CLI we can be agnostic about what
they really do.

In theory, each model should implement the following predictions:

* vacuum tresp
* vacuum transition dipole
* environment tresp
* environment transition dipole
* vacuum site energy
* environment electrochromic shift
* environment site energy
* polarizable LR contribution
* polarizable site energy

At the end, make the model available in the dictionary below
("available_models").
"""

from .jctc2023 import Model_JCTC2023


available_models = {
    "CLA": {"JCTC2023": Model_JCTC2023()},
    "CHL": {"JCTC2023": Model_JCTC2023()},
    # the JCTC2023 model for BCL is called like this because it
    # has the same structure, just different parameters.
    "BCL": {"JCTC2023": Model_JCTC2023()},
}
