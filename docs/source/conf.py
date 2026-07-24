import os
from os.path import relpath, dirname
from pathlib import Path
import re
import sys
import warnings
from datetime import date

sys.path.insert(0, str(Path('..', '..', 'src', 'python').resolve()))


import spatialize

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Spatialize'
copyright = f'2024-{date.today().year}, ALGES Lab'
author = 'ALGES Lab'
version = re.sub(r'\.dev.*$', r'.dev', spatialize.__version__)
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "numpydoc",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.ifconfig",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    # "sphinx.ext.linkcode",
    'sphinx_math_dollar',
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    # "nbsphinx",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -----------------------------------------------------------------------------
# Autodoc
# -----------------------------------------------------------------------------

autodoc_mock_imports = ['libspatialize', 'cv2']
autodoc_default_options = {
    'members': True
}
autodoc_docstring_signature = True

# -----------------------------------------------------------------------------
# Intersphinx
# -----------------------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

# -----------------------------------------------------------------------------
# Nitpick ignore
# -----------------------------------------------------------------------------
# References we have deliberately chosen not to document (internal logging
# callbacks, the SpatializeError exception), plus a handful of docstring-level
# cross-references we cannot fix without editing docstring content (out of
# scope for this pass): references that use a private/internal import path
# instead of the public re-export path we document under, a Sphinx field-list
# ":type ...: ..., optional" docstring that trips typehint parsing on the bare
# word "optional", a reference to a private undocumented base class, an
# autosummary-generated link to a third-party callable's __call__, and one
# :func: role used where :meth: was meant for a matplotlib method. Kept
# minimal: every entry here is a dangling reference we accept, not a
# workaround for something that should otherwise resolve once the reference
# pages exist.
nitpick_ignore = [
    ("py:func", "spatialize.logging.default_singleton_callback"),
    ("py:func", "spatialize.logging.singleton_null_callback"),
    ("py:exc", "spatialize.SpatializeError"),
    ("py:class", "spatialize.SpatializeError"),
    # Private base class of EmpiricalModel, intentionally undocumented.
    ("py:obj", "BaseEmpiricalModel"),
    # Autosummary-generated link for a third-party callable attribute.
    ("py:obj", "spatialize.empirical.EmpiricalModel.F.__call__"),
]

# -----------------------------------------------------------------------------
# numpydoc
# -----------------------------------------------------------------------------
numpydoc_show_class_members = True
numpydoc_show_inherited_class_members = False
numpydoc_attributes_as_param_list = False
numpydoc_class_members_toctree = False

# -----------------------------------------------------------------------------
# Autosummary
# -----------------------------------------------------------------------------

autosummary_generate = True

# -----------------------------------------------------------------------------
# Coverage checker
# -----------------------------------------------------------------------------
coverage_ignore_modules = r"""
    """.split()
coverage_ignore_functions = r"""
    test($|_) (some|all)true bitwise_not cumproduct pkgload
    generic\.
    """.split()
coverage_ignore_classes = r"""
    """.split()

coverage_c_path = []
coverage_c_regexes = {}
coverage_ignore_c_items = {}


# -----------------------------------------------------------------------------
# HTML output
# -----------------------------------------------------------------------------

pygments_style = "sphinx"

#html_theme = 'alabaster'
html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
# Drop Sphinx's default "<project> vX.Y documentation" long form in the
# sidebar/browser-tab title; just "Spatialize <version>".
html_title = f"Spatialize {version}"
html_theme_options = {
  "show_nav_level": 2,
  "navbar_end": ["navbar-icon-links"],
  "header_links_before_dropdown": 5,
  "navigation_with_keys": True,
  # Only GitHub here (repo-specific target) — kept identical in shape to
  # spatialize-examples/docs/conf.py, which has the same single GitHub icon
  # pointing at its own repo.
  "icon_links": [
      {
          "name": "GitHub",
          "url": "https://github.com/alges/spatialize",
          "icon": "fa-brands fa-github",
      },
  ],
}

# Top navbar links (Home, Documentation, Examples) are rendered by
# docs/source/_templates/navbar-nav.html, which overrides the theme's stock
# component (see that file for why: it avoids flooding the header with a
# link per hidden-toctree entry, and uses `pathto()` for same-project links
# so they resolve correctly at any page depth without leaving this site).
