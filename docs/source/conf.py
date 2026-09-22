from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

project = "memento"
copyright = "2026, Min Cheol Kim"
author = "Min Cheol Kim"
release = "0.1.3"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
]
exclude_patterns = []
html_theme = "sphinx_rtd_theme"
