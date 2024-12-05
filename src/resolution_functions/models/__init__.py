"""
Package containing the implementation of the various :term:`models<model>`.

All classes within are exposed for reference only and should not be instantiated directly. For
obtaining the :term:`resolution function` of an :term:`instrument`, please use the
`resolution_functions.instrument.Instrument.get_resolution_function` method.

Advanced Use
------------
For work with custom models, see the `model_base` module, but do consider contributing any work.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .polynomial import PolynomialModel1D, DiscontinuousPolynomialModel1D
from .panther_abins import PantherAbINSModel
from .pychop import PyChopModelFermi, PyChopModelNonFermi, PyChopModelCNCS, PyChopModelLET
from .tosca_book import ToscaBookModel
from .vision_paper import VisionPaperModel

if TYPE_CHECKING:
    from .model_base import InstrumentModel


MODELS: dict[str, type[InstrumentModel]] = {
    'polynomial_1d': PolynomialModel1D,
    'discontinuous_polynomial': DiscontinuousPolynomialModel1D,
    'tosca_book': ToscaBookModel,
    'vision_paper': VisionPaperModel,
    'panther_abins_polynomial': PantherAbINSModel,
    'pychop_fit_fermi': PyChopModelFermi,
    'pychop_fit_cncs': PyChopModelCNCS,
    'pychop_fit_let': PyChopModelLET,
}
"""A dictionary mapping the unique name of a model to the corresponding class."""
