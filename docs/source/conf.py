# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'resolution_functions'
copyright = '2024, Rastislav Turanyi, Adam J Jackson'
author = 'Rastislav Turanyi, Adam J Jackson'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # 'sphinx.ext.napoleon',
    'numpydoc',
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    # 'sphinx.ext.autosummary',
    'myst_parser',
    # 'sphinx_autodoc_typehints',
]

templates_path = ['_templates']
exclude_patterns = []

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None)
}

master_doc = 'index' # Otherwise readthedocs searches for contents.rst

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']


# -- Options for napoleon -------------------------------------------------

# napoleon_google_docstring = False
# napoleon_numpy_docstring = True
# napoleon_include_init_with_doc = False
# napoleon_include_special_with_doc = True
# napoleon_attr_annotations = True

numpydoc_xref_param_type = True

# The reST default role (used for this markup: `text`) to use for all
# documents.
default_role = "obj"
