import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------
project = 'memento'
copyright = '2025, Min Cheol Kim'
author = 'YMin Cheol Kim'
release = '0.1.1'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages'
]

# Paths that contain templates.
templates_path = ['_templates']

# Exclude patterns when looking for source files.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
