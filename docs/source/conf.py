# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

project = "excipy"
copyright = "2022, excipy authors"
author = (
    "Edoardo Cignoni, Elena Betti, Chris John, Lorenzo Cupellini, Benedetta Mennucci"
)

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
# html_sidebars = {
#     '**': [
#         'about.html',
#         'navigation.html',
#         'relations.html',
#         'searchbox.html',
#         'donate.html',
#     ]
# }

# https://alabaster.readthedocs.io/en/latest/customization.html
html_theme_options = {
    "logo": "../../../images/logo.svg",
    "logo_name": True,
    "description": "Machine learning models for a fast estimation of excitonic Hamiltonians",
    "github_user": "Molecolab-Pisa",
    "github_repo": "excipy",
    "github_button": True,
    "page_width": "90%",
    "sidebar_width": "25%",
    "anchor": "#689913",
    "link": "#689913",
    "font_size": "20px",
}
html_show_sphinx = True
html_static_path = ["_static"]
