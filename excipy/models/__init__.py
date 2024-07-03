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
