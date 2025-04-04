# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
from __future__ import annotations

import nbconvert
import nbformat

# Read the demo notebook and write it to RST
demo_notebook = nbformat.read("examples/demo.ipynb", as_version=4)
rst_exporter = nbconvert.RSTExporter()
# rst_exporter.template_file = 'rst'
demo_rst, resources = rst_exporter.from_notebook_node(demo_notebook)

# Write the RST to a file
with open("examples/demo.rst", "w") as f:
    f.write(demo_rst)

# Write the resources to a file
for key, val in resources["outputs"].items():
    with open(f"examples/{key}", "wb") as f:
        f.write(val)


# -- Project information -----------------------------------------------------

project = "pyINSPECTA"
copyright = "2023, CSIRO"
author = "Lawrence Toomey, George Hobbs, Alec Thomson"

# The full version, including alpha/beta/rc tags
release = "0.1.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "autoapi.extension",
    "sphinx_mdinclude",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

autoapi_dirs = ["../pyINSPECTA"]


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
