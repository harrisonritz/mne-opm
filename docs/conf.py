"""Sphinx configuration for mne-opm documentation."""

import sys
from pathlib import Path

# -- Path setup ---------------------------------------------------------------
# Add src/ so that autodoc can import custom.preprocessing.*
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# -- Project information ------------------------------------------------------
project = "mne-opm"
copyright = "2025, Harrison Ritz"
author = "Harrison Ritz"
version = "0.1.0"
release = version

# -- General configuration ----------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

# MyST-Parser settings (Markdown support)
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
]

# Source file suffixes
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# The master toctree document
master_doc = "index"

# Patterns to exclude when looking for source files
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Suppress known harmless warnings:
#   - intersphinx network failures (offline/CI environments)
#   - duplicate IDs from repeated section headers in module docstrings
#   - autodoc duplicate object descriptions from inherited class attributes
suppress_warnings = [
    "intersphinx.external",
    "autodoc.duplicate_object",
]

# Do not treat docutils errors (duplicate section IDs in docstrings) as fatal.
# These come from module-level docstrings that contain RST section headers like
# "Usage" and "Configuration Attributes" across multiple modules.
nitpicky = False

# -- Autodoc settings ---------------------------------------------------------
# Mock imports for heavy scientific packages so docs build without them.
autodoc_mock_imports = [
    "mne",
    "mne_bids",
    "mne_bids_pipeline",
    "mne_qt_browser",
    "osl_ephys",
    "numpy",
    "scipy",
    "matplotlib",
    "sklearn",
    "pandas",
    "pyvista",
    "vtk",
    "openmeeg",
    "nibabel",
    "nilearn",
    "trame",
    "dcm2niix",
    "dupy",
]
autodoc_member_order = "bysource"
autodoc_typehints = "description"

# -- Napoleon settings (NumPy-style docstrings) --------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# -- Intersphinx ---------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "mne": ("https://mne.tools/stable/", None),
    "mne_bids": ("https://mne.tools/mne-bids/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# -- HTML output ---------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = "mne-opm"
