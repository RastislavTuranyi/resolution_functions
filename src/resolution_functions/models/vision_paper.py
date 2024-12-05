"""
Model for TOSCA-like :term:`instruments<instrument>` from the [VISION-paper]_.

All classes within are exposed for reference only and should not be instantiated directly. For
obtaining the :term:`resolution function` of an :term:`instrument`, please use the
`resolution_functions.instrument.Instrument.get_resolution_function` method.

.. [VISION-paper] https://doi.org/10.1016/j.nima.2009.03.204
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import numpy as np

from .model_base import InstrumentModel, ModelData

if TYPE_CHECKING:
    from jaxtyping import Float


@dataclass(init=True, repr=True, frozen=True, slots=True, kw_only=True)
class VisionPaperModelData(ModelData):
    """
    Data for the `VisionPaperModel` :term:`model`.

    Attributes
    ----------
    function
        The name of the function, i.e. the alias for `PantherAbINSModel`.
    citation
        The citation for the model. Please use this to look up more details and cite the model.
    primary_flight_path
        Distance between the :term:`moderator` and the :term:`sample` in meters (m).
    primary_flight_path_uncertainty
        The uncertainty associated with the `primary_flight_path`, in meters (m).
    sample_thickness
        Thickness of the :term:`sample` in meters (m).
    detector_thickness
        Thickness of the :term:`detector` in meters (m).
    crystal_plane_spacing
        Distance between the layers of atoms making up the :term:`detector`, in meters (m).
    d_r
        Uncertainty associated with the :term:`detector` offset, in meters (m).
    d_t
        Uncertainty associated with the :term:`source` pulse shape, in ? (?).
    angles
        Angle between the :term:`sample` and the analyser, in degrees.
    distance_sample_analyzer
        Distance between the :term:`sample` and the analyser, in meters (m).
    average_bragg_angle_graphite
        Average Bragg angle of the graphite analyser, in degrees.
    restrictions
    defaults
    """
    primary_flight_path: float
    primary_flight_path_uncertainty: float
    sample_thickness: float
    detector_thickness: float
    crystal_plane_spacing: float
    d_r: float
    d_t: float
    angles: list[float]
    distance_sample_analyzer: float
    average_bragg_angle_graphite: float


class VisionPaperModel(InstrumentModel):
    """
    Model for TOSCA-like :term:`instruments<instrument>` from the [VISION paper]_.

    Models the :term:`resolution` as a function of energy transfer (frequencies) only, with the
    output model being an Ikeda-Carpenter distribution. This is done by taking into account the
    contributions from the various parts of the :term:`instrument` (for more information, please see
    the reference).

    Parameters
    ----------
    model_data
        The data associated with the model for a given version of a given instrument.

    Attributes
    ----------
    input
        The input that the ``__call__`` method expects.
    output
        The output of the ``__call__`` method.
    data_class
        Reference to the `VisionPaperModelData` type.
    citation
    """
    input = 1
    output = 1

    data_class = VisionPaperModelData

    PLANCK = 6.626068e-34  # J s
    REDUCED_PLANCK = 1.054571817e-34  # J s
    NEUTRON_MASS = 1.67492749804e-27  # kg

    def __init__(self, model_data: VisionPaperModelData, **_):
        super().__init__(model_data)

        self.l1 = model_data.primary_flight_path
        self.d_t = model_data.d_t

        self.e0 = self.PLANCK ** 2 * 0.5 / self.NEUTRON_MASS * (0.5 / model_data.crystal_plane_spacing) ** 2
        self.nu0 = 0.5 * self.PLANCK / (self.NEUTRON_MASS * model_data.crystal_plane_spacing)
        self.one_over_l1 = 1 / self.l1
        self.distance_ratio = model_data.primary_flight_path_uncertainty * self.one_over_l1

        self.theta = np.deg2rad(model_data.average_bragg_angle_graphite)
        self.capital_t = 0.5 * 1 / np.tan(self.theta)

        self.z2 = model_data.distance_sample_analyzer
        self.capital_t_over_z2 = self.capital_t / self.z2

        self.d_a = model_data.sample_thickness ** 2 / 12
        d_b = 0.7e-6
        d_c = model_data.detector_thickness ** 2 / 12
        self.db_dc_factor = (2 * d_b + d_c)

        self.final_term = self.e0 / np.tan(self.theta) / self.z2 * model_data.d_r

    def __call__(self, frequencies: Float[np.ndarray, 'frequencies']) -> Float[np.ndarray, 'sigma']:
        """
        Evaluates the model at given energy transfer values (`frequencies`), returning the
        corresponding Ikeda-Carpenter widths (FWHM).

        Parameters
        ----------
        frequencies
            Energy transfer in meV. The frequencies at which to return widths.

        Returns
        -------
        fwhm
            The Ikeda-Carpenter widths at `frequencies` as predicted by this model.
        """
        e1 = frequencies * self.REDUCED_PLANCK + self.e0 * (1 / np.sin(self.theta))
        z0 = self.l1 * (self.e0 / e1) ** 0.5
        one_over_z0 = 1 / z0

        sigma = self.distance_ratio - self.nu0 * self.d_t / z0
        sigma += (self.one_over_l1 + one_over_z0 + self.capital_t_over_z2) * self.d_a
        sigma += (one_over_z0 + self.capital_t_over_z2) * self.db_dc_factor
        sigma *= 2 * e1
        sigma -= self.final_term

        return sigma
