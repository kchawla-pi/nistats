"""
functional MRI module for NeuroImaging in python
--------------------------------------------------

Documentation is available in the docstrings and online at
http://nistats.github.io.

Contents
--------
Nistats is a Python module for fast and easy functional MRI statistical
analysis.

Submodules
---------
datasets                --- Utilities to download NeuroImaging datasets
hemodynamic_models      --- Hemodyanmic response function specification
design_matrix           --- Design matrix creation for fMRI analysis
experimental_paradigm   --- Experimental paradigm files checks and utils
model                   --- Statistical tests on likelihood models
regression              --- Standard regression models
first_level_model       --- API for first level fMRI model estimation
second_level_model      --- API for second level fMRI model estimation
contrasts               --- API for contrast computation and manipulations
thresholding            --- Utilities for cluster-level statistical results
reporting               --- Utilities for creating reports & plotting data
utils                   --- Miscellaneous utilities
"""

import sys
import warnings

from .version import _check_module_dependencies, __version__


def _library_deprecation_warning():
    lib_dep_warning = (
        'Starting with Nilearn 0.7.0, all Nistats functionality '
        'has been incorporated into Nilearn. '
        'Nistats package will no longer be updated or maintained.')
    warnings.filterwarnings('once', message=lib_dep_warning)
    warnings.warn(message=lib_dep_warning,
                  category=FutureWarning,
                  stacklevel=3)

_check_module_dependencies()
_library_deprecation_warning()

__all__ = ['__version__', 'datasets', 'design_matrix']
