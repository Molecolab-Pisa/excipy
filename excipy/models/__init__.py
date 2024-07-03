from .jctc2023 import Model_JCTC2023

available_models = {
    "CLA": {"JCTC2023": Model_JCTC2023()},
    "CHL": {"JCTC2023": Model_JCTC2023()},
    # the JCTC2023 model for BCL is called like this because it
    # has the same structure, just different parameters.
    "BCL": {"JCTC2023": Model_JCTC2023()},
}
