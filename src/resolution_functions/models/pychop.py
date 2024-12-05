"""
The [PyChop]_ :term:`model`, a 1D :term:`model` for direct-geometry 2D
:term:`instruments<instrument>`.

[PyChop]_, originating from [Mantid]_ as a single :term:`model`, is here organised as a collection
of models. These can be split into two types: the `PyChopModelFermi` :term:`model`, used for all
:term:`instruments<instrument>` with a :term:`Fermi chopper` (which is the sole determinant of the
:term:`chopper` contribution to the :term:`resolution`), and models such as `PyChopModelLET` and
`PyChopModelCNCS`, which are models of instruments without a :term:`Fermi chopper`, in which the
first and last :term:`choppers<chopper>` determine the :term:`chopper` contribution to the
:term:`resolution`.

All classes within are exposed for reference only and should not be instantiated directly. For
obtaining the :term:`resolution function` of an :term:`instrument`, please use the
`resolution_functions.instrument.Instrument.get_resolution_function` method.

.. [PyChop] https://github.com/mducle/pychop/tree/main
.. [Mantid] https://mantidproject.org/
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from copy import deepcopy
from math import erf
from typing import Optional, TYPE_CHECKING, Union

try:
    from typing import NotRequired, TypedDict
except ImportError:
    from typing_extensions import NotRequired, TypedDict


import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy.interpolate import interp1d

from .model_base import InstrumentModel, ModelData, InvalidInputError

if TYPE_CHECKING:
    from jaxtyping import Float

E2L = 81.8042103582802156
E2V = 437.393362604208619
E2K = 0.48259640220781652
SIGMA2FWHM = 2 * np.sqrt(2 * np.log(2))
SIGMA2FWHMSQ = SIGMA2FWHM**2


@dataclass(init=True, repr=True, frozen=True, slots=True)
class PyChopModelData(ModelData):
    """
    Base class for PyChop :term:`model` data.

    Corresponds to the abstract base class `PyChopModel` and so should not be used directly.
    Concrete implementations of the PyChop model should have their own model data subclasses of this
    class.

    Attributes
    ----------
    function
        The name of the function, i.e. the alias for `PantherAbINSModel`.
    citation
        The citation for the model. Please use this to look up more details and cite the model.
    d_chopper_sample
        Distance from the final chopper to sample in meters (m).
    d_sample_detector
        Distance from sample to detector in meters (m).
    aperture_width
        Width of aperture at moderator face in meters (m)
    theta
        The angle that the beamline makes with the moderator face in degrees.
    default_e_init
        The default value for the initial energy (``e_init``) in meV.
    allowed_e_init
        The limits (lower and upper bound) for the allowed initial energies (``e_init``). This
        should correspond to the physical limitations of the INS instrument.
    frequency_matrix
        A matrix mapping the relationship between the user-provided parameter ``chopper_frequency``
        or its equivalent (depending on model) to the frequency of each chopper in the instrument.
    choppers
        Data for each chopper in the instrument. See `FermiChopper` and `DiskChopper` for more info.
    moderator
        Data for the moderator. See `Moderator` for more info.
    detector
        Data for the detector. See `Detector` for more info.
    sample
        Data for the sample. See `Sample` for more info.
    tjit
        The jitter time in microseconds (us).
    restrictions
    defaults
    """
    d_chopper_sample: float
    d_sample_detector: float
    aperture_width: float
    theta: float
    default_e_init: float
    allowed_e_init: list[float]
    frequency_matrix: list[list[float]]
    choppers: dict[str, FermiChopper | DiskChopper]
    moderator: Moderator
    detector: None | Detector
    sample: None | Sample
    tjit: float


@dataclass(init=True, repr=True, frozen=True, slots=True)
class PyChopModelDataFermi(PyChopModelData):
    """
    Data for the `PyChopModelFermi` :term:`model`.

    Attributes
    ----------
    function
        The name of the function, i.e. the alias for `PantherAbINSModel`.
    citation
        The citation for the model. Please use this to look up more details and cite the model.
    d_chopper_sample
        Distance from the final :term:`chopper` to :term:`sample` in meters (m).
    d_sample_detector
        Distance from :term:`sample` to :term:`detector` in meters (m).
    aperture_width
        Width of aperture at :term:`moderator` face in meters (m)
    theta
        The angle that the beamline makes with the :term:`moderator` face in degrees.
    default_e_init
        The default value for the initial energy (``e_init``) in meV.
    allowed_e_init
        The limits (lower and upper bound) for the allowed initial energies (``e_init``). This
        should correspond to the physical limitations of the INS :term:`instrument`.
    frequency_matrix
        A matrix mapping the relationship between the user-provided parameter ``chopper_frequency``
        to the frequency of each :term:`chopper` in the :term:`instrument`.
    choppers
        Data for each :term:`chopper` in the :term:`instrument`. See `FermiChopper` and `DiskChopper` for more info.
    moderator
        Data for the :term:`moderator`. See `Moderator` for more info.
    detector
        Data for the :term:`detector`. See `Detector` for more info.
    sample
        Data for the :term:`sample`. See `Sample` for more info.
    tjit
        The jitter time in microseconds (us).
    default_chopper_frequency
        The default value for the :term:`Fermi chopper` frequency (``chopper_frequency``) in Hz.
    allowed_chopper_frequencies
        The allowed values for the :term:`Fermi chopper` frequency (``chopper_frequency``) in Hz.
        `allowed_chopper_frequencies` defines the (start, stop, step), which is converted into a
        list at runtime.
    pslit
        Width of the neutron-transparent slit in meters (m).
    radius
        Radius of the :term:`chopper` package in meters (m).
    rho
        Curvature of the :term:`chopper` package in meters (m).
    restrictions
    defaults
    """
    default_chopper_frequency: int
    allowed_chopper_frequencies: list[int]
    pslit: float
    radius: float
    rho: float

    @property
    def restrictions(self) -> dict[str, list[int | float]]:
        return {'e_init': self.allowed_e_init, 'chopper_frequency': self.allowed_chopper_frequencies}

    @property
    def defaults(self) -> dict:
        return {'e_init': self.default_e_init, 'chopper_frequency': self.default_chopper_frequency}


@dataclass(init=True, repr=True, frozen=True, slots=True)
class PyChopModelDataNonFermi(PyChopModelData):
    """
    Data for the PyChop :term:`models<model>` that do not have a :term:`Fermi chopper` :

    - `PyChopModelCNCS`
    - `PyChopModelLET`

    Attributes
    ----------
    function
        The name of the function, i.e. the alias for `PantherAbINSModel`.
    citation
        The citation for the model. Please use this to look up more details and cite the model.
    d_chopper_sample
        Distance from the final :term:`chopper` to :term:`sample` in meters (m).
    d_sample_detector
        Distance from :term:`sample` to :term:`detector` in meters (m).
    aperture_width
        Width of aperture at :term:`moderator` face in meters (m)
    theta
        The angle that the beamline makes with the :term:`moderator` face in degrees.
    default_e_init
        The default value for the initial energy (``e_init``) in meV.
    allowed_e_init
        The limits (lower and upper bound) for the allowed initial energies (``e_init``). This
        should correspond to the physical limitations of the INS :term:`instrument`.
    frequency_matrix
        A matrix mapping the relationship between all user-provided :term:`chopper` frequency
        parameters (i.e. the choppers with user control) to the frequency of each :term:`chopper` in
        the :term:`instrument`.
    choppers
        Data for each :term:`chopper` in the :term:`instrument`. See `DiskChopper` for more info.
    moderator
        Data for the :term:`moderator`. See `Moderator` for more info.
    detector
        Data for the :term:`detector`. See `Detector` for more info.
    sample
        Data for the :term:`sample`. See `Sample` for more info.
    tjit
        The jitter time in microseconds (us).
    default_chopper_frequency
        The default value for the :term:`chopper` frequency of each user-controlled :term:`chopper`
        in Hz.
    allowed_chopper_frequencies
        The allowed values for the :term:`chopper` frequency of each user-controlled
        :term:`chopper`, in Hz. Each value defines the (start, stop, step), which is converted into
        a list at runtime.
    constant_frequencies
        The frequency of each :term:`chopper` in Hz, with those run at a constant frequency having
        non-zero values.
    source_frequency
        The frequency of the neutron :term:`source` in Hz.
    n_frame
        Number of frames to calculate time-distance diagram for.
    restrictions
    defaults
    """
    default_chopper_frequency: list[int]
    allowed_chopper_frequencies: list[list[int]]
    constant_frequencies: list[int]
    source_frequency: float
    n_frame: int

    @property
    def restrictions(self) -> dict[str, list[int | float]]:
        return {'e_init': self.allowed_e_init, 'chopper_frequency': self.allowed_chopper_frequencies}

    @property
    def defaults(self) -> dict:
        return {'e_init': self.default_e_init, 'chopper_frequency': self.default_chopper_frequency}


class FermiChopper(TypedDict):
    """
    Data for a :term:`Fermi chopper`.

    Attributes
    ----------
    distance
        Distance from :term:`moderator` to this :term:`Fermi chopper` in meters (m).
    aperture_distance
        Distance from aperture (:term:`moderator` face) to this :term:`Fermi chopper` in meters (m).
    """
    distance: float
    aperture_distance: float


class DiskChopper(TypedDict):
    """
    Data for a :term:`disk chopper`.

    Attributes
    ----------
    distance
        Distance from :term:`moderator` to this :term:`disk chopper` in meters (m).
    nslot
        Number of slots in the :term:`chopper`.
    slot_width
        Width of all slots (assumed to be equal) in millimeters (mm).
    slot_ang_pos
        Angular position of each slot in the :term:`chopper`, in degrees. Must be monotonically
        increasing. If None, equal spacing is assumed.
    guide_width
        Width of the guide after the :term:`chopper` in millimeters (mm).
    radius
        Disk radius in millimeters (mm)?
    num_disk
        Number of disks making up this :term:`disk chopper`.
    is_phase_independent
        Whether this disk is supposed to be phased independently.
    default_phase
        Default phase for this :term:`disk chopper`. If an ``int`` is provided, it is assumed to be
        time in microseconds (us), and if a ``str`` is provided, it is assumed to be a slot index
        for the desired rep to go through.
    """
    distance: float
    nslot: int
    slot_width: float
    slot_ang_pos: Union[list[float], None]
    guide_width: float
    radius: float
    num_disk: int
    is_phase_independent: bool
    default_phase: Union[int, str]


class Sample(TypedDict):
    """
    Data for the :term:`sample`.

    Attributes
    ----------
    type
        Sample type: 0==flat plate, 1==ellipse, 2==annulus, 3==sphere, 4==solid cylinder.
    thickness
        Sample thickness in meters (m).
    width
        Sample width in meters (m).
    height
        Sample height in meters (m).
    gamma
        Angle of x-axis to ki in degrees.
    """
    type: int
    thickness: float
    width: float
    height: float
    gamma: float
    
    
class Detector(TypedDict):
    """
    Data for the :term:`detector`.

    Attributes
    ----------
    type
        Detector type: 1==Li Davidson scintillator, 2==Helium tube binned together, 3==Helium tube.
    phi
        Detector scattering angle in degrees.
    depth
        Detector depth (diameter for tube) in meters (m).
    """
    type: int
    phi: float
    depth: float


class Moderator(TypedDict):
    """
    Data for the :term:`moderator`.

    Attributes
    ----------
    type
        Moderator time profile type: 0==chi squared, 1==Ikeda-Carpenter, 2==modified chi squared,
        3==polynomial.
    parameters
        Time profile parameters.
    scaling_function
        The name of the scaling function to use. If None, no scaling will be applied.
    scaling_parameters
        The parameters to use for the `scaling_function`. Does not have to be provided if
        `scaling_function` is None.
    measured_wavelength
        The wavelength at which the moderator contribution to the resolution was measured
        experimentally. If None, interpolation will not be used and instead the moderator
        contribution to the resolution will be computed analytically.
    measured_width
        The experimentally measured moderator contribution to the resolution (width of the Gaussian)
        at the `measured_wavelength`. Does not have to be provided if `measured_wavelength` is None.
    """
    type: int
    parameters: list[float]
    scaling_function: None | str
    scaling_parameters: NotRequired[list[float]]
    measured_wavelength: None | list[float]
    measured_width: NotRequired[list[float]]


class PyChopModel(InstrumentModel, ABC):
    """
    Abstract base class for all PyChop models.

    This class provides the concrete implementation for most of PyChop algorithm, only the chopper
    contribution to the resolution is missing. The following methods must be implemented:

    - ``__init__`` which must take the ``model_data`` and ``e_init`` parameters as well as some form
      of chopper frequency parameters. It should fit a polynomial to the data obtained from the
      `_precompute_resolution` method. It must also use the provided `_validate_e_init` method.
    - `polynomial` - this property should be implemented in such a way that it exposes the
      polynomial computed in ``__init__``.
    - `_get_chopper_width_squared` which calculates the chopper contribution.

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
        Reference to the `PyChopModelData` type.
    citation
    polynomial
    """
    input = 1
    output = 1

    data_class = PyChopModelData

    def __call__(self, frequencies: Float[np.ndarray, 'frequencies'], *args, **kwargs
                 ) -> Float[np.ndarray, 'sigma']:
        """
        Evaluates the model at given energy transfer values (`frequencies`), returning the
        corresponding Gaussian widths (sigma).

        Parameters
        ----------
        frequencies
            Energy transfer in meV. The frequencies at which to return widths.

        Returns
        -------
        sigma
            The Gaussian widths at `frequencies` as predicted by this model.
        """
        return self.polynomial(frequencies)

    @property
    @abstractmethod
    def polynomial(self) -> Polynomial:
        """
        The polynomial fitted to the resolutions.

        Returns
        -------
        polynomial
            The polynomial modelling the resolution of an instrument at a particular value of
            incident energy and chopper frequency.
        """
        pass

    @staticmethod
    def _validate_e_init(e_init: float | None, model_data: PyChopModelData) -> float:
        """
        Validates that the user-provided `e_init` is within the allowed range for this instrument.

        Parameters
        ----------
        e_init
            The user-provided incident energy. If None, the default value for this instrument is
            used instead.
        model_data
            The data for a particular INS instrument.

        Returns
        -------
        e_init
            Valid incident energy.

        Raises
        ------
        InvalidInputError
            If the provided `e_init` is not within the range allowed for the instrument.
        """
        if e_init is None:
            e_init = model_data.default_e_init
        elif not model_data.allowed_e_init[0] <= e_init <= model_data.allowed_e_init[1]:
            raise InvalidInputError(f'The provided incident energy ({e_init}) is not allowed; only values within the '
                                    f'range of {model_data.allowed_e_init} are possible.')

        return e_init

    @classmethod
    def _precompute_resolution(cls,
                               model_data: PyChopModelData,
                               e_init: float,
                               chopper_frequencies: list[int]
                               ) -> tuple[Float[np.ndarray, 'frequency'],
                                          Float[np.ndarray, 'resolution']]:
        """
        Calculates the resolution on a coarse grid.

        The grid consists of 40 equally spaced values between 0 meV and `e_init` meV.

        Parameters
        ----------
        model_data
            The data for a particular version of a particular instrument.
        e_init
            The initial energy, as selected by the user.
        chopper_frequencies
            The frequency of each user-controlled chopper.

        Returns
        -------
        fake_frequencies
            The grid of energy transfer (``frequencies``) on which the resolution was evaluated.
        resolution
            The resolution at `fake_frequencies` in sigma (Gaussian width).

        Raises
        ------
        NoTransmissionError
            If the instrument contains a Fermi chopper, and the combination of `e_init` and
            `chopper_frequency` would result in the Fermi chopper blocking the neutron beam,
            resulting in no signal.
        """
        fake_frequencies = np.linspace(0, e_init, 40, endpoint=False)
        vsq_van = cls._precompute_van_var(model_data, e_init, chopper_frequencies, fake_frequencies)
        e_final = e_init - fake_frequencies
        resolution = (2 * E2V * np.sqrt(e_final ** 3 * vsq_van)) / model_data.d_sample_detector

        return fake_frequencies, resolution / SIGMA2FWHM

    @classmethod
    def _precompute_van_var(cls,
                            model_data: PyChopModelData,
                            e_init: float,
                            chopper_frequencies: list[int],
                            fake_frequencies: Float[np.ndarray, 'frequency'],
                            ) -> Float[np.ndarray, 'resolution']:
        """
        Calculates the time squared FWHM in s^2 at the sample for all components.

        Parameters
        ----------
        model_data
            The data for a particular version of a particular instrument.
        e_init
            The initial energy, as selected by the user.
        chopper_frequencies
            The frequency of each user-controlled chopper.
        fake_frequencies
            The coarse grid of energy transfer at which to evaluate the contributions to resolution.

        Returns
        -------
        vsq_van
            The time squared FWHM in s^2 at the sample.

        Raises
        ------
        NoTransmissionError
            If the instrument contains a Fermi chopper, and the combination of `e_init` and
            `chopper_frequency` would result in the Fermi chopper blocking the neutron beam,
            resulting in no signal.
        """
        tsq_jit = model_data.tjit ** 2
        x0, xa, xm = cls._get_distances(model_data.choppers)
        x1, x2 = model_data.d_chopper_sample, model_data.d_sample_detector

        tsq_moderator = cls._get_moderator_width_squared(model_data.moderator, e_init)
        tsq_chopper = cls._get_chopper_width_squared(model_data, e_init, chopper_frequencies)

        # For Disk chopper spectrometers, the opening times of the first chopper can be the effective moderator time
        if tsq_chopper[1] is not None:
            frac_dist = 1 - (xm / x0)
            tsmeff = tsq_moderator * frac_dist ** 2  # Effective moderator time at first chopper
            x0 -= xm  # Propagate from first chopper, not from moderator (after rescaling tmod)
            tsq_moderator = tsmeff if (tsq_chopper[1] > tsmeff) else tsq_chopper[1]

        tsq_chopper = tsq_chopper[0]
        tanthm = np.tan(np.deg2rad(model_data.theta))
        omega = chopper_frequencies[0] * 2 * np.pi

        vi = E2V * np.sqrt(e_init)
        vf = E2V * np.sqrt(e_init - fake_frequencies)
        vratio = (vi / vf) ** 3

        factor = omega * (xa + x1)
        g1 = (1.0 - (omega * tanthm / vi) * (xa + x1))
        f1 = (1.0 + (x1 / x0) * g1) / factor
        g1 /= factor

        modfac = (x1 + vratio * x2) / x0
        chpfac = 1.0 + modfac
        apefac = f1 + ((vratio * x2 / x0) * g1)

        tsq_moderator *= modfac ** 2
        tsq_chopper *= chpfac ** 2
        tsq_jit *= chpfac ** 2
        tsq_aperture = apefac ** 2 * (model_data.aperture_width ** 2 / 12.0) * SIGMA2FWHMSQ

        vsq_van = tsq_moderator + tsq_chopper + tsq_jit + tsq_aperture

        if model_data.detector is not None:
            tsq_detector = (1. / vf) ** 2 * cls._get_detector_width_squared(model_data.detector, fake_frequencies, e_init)
            vsq_van += tsq_detector

            phi = np.deg2rad(model_data.detector['phi'])
        else:
            phi = 0.

        if model_data.sample is not None:
            g2 = (1.0 - (omega * tanthm / vi) * (x0 - xa))
            f2 = (1.0 + (x1 / x0) * g2) / factor
            g2 /= factor

            gamma = np.deg2rad(model_data.sample['gamma'])
            bb = - np.sin(gamma) / vi + np.sin(gamma - phi) / vf - f2 * np.cos(gamma)
            sample_factor = bb - (vratio * x2 / x0) * g2 * np.cos(gamma)

            tsq_sample = sample_factor ** 2 * cls._get_sample_width_squared(model_data.sample)
            vsq_van += tsq_sample

        # return vsq_van, tsq_moderator, tsq_chopper, tsq_jit, tsq_aperture, tsq_detector, tsq_sample
        # return vsq_van, tsq_moderator, tsq_chopper, tsq_jit, tsq_aperture
        return vsq_van

    @classmethod
    def _get_moderator_width_squared(cls,
                                     moderator_data: Moderator,
                                     e_init: float, ):
        """
        Calculates the moderator contribution to the resolution squared, in FWHM.

        If the moderator data contains the experimentally measured widths, and the `e_init` lies
        within the measured range of wavelengths, the result is obtained via interpolation of the
        measured widths. Otherwise, the result is computed analytically.

        Parameters
        ----------
        moderator_data
            The data for the moderator.
        e_init
            The initial energy, as selected by the user.

        Returns
        -------
        moderator_width_squared
            The moderator contribution to the resolution squared, in FWHM.

        Raises
        ------
        NotImplementedError
            If analytical and the moderator type is not one of the defined values.

        See Also
        --------
        _get_moderator_width_analytical : Analytical computation of the moderator contribution.
        """
        wavelengths = moderator_data['measured_wavelength']
        if wavelengths is not None:
            # TODO: Sort the data in the yaml file and remove sorting below
            wavelengths = np.array(wavelengths)
            idx = np.argsort(wavelengths)
            wavelengths = wavelengths[idx]
            widths = np.array(moderator_data['measured_width'])[idx]

            interpolated_width = interp1d(wavelengths, widths, kind='slinear')

            wavelength = np.sqrt(E2L / e_init)
            if wavelength >= wavelengths[0]:  # Larger than the smallest OG value
                width = interpolated_width(min([wavelength, wavelengths[-1]])) / 1e6  # Table has widths in microseconds
                return width ** 2  # in FWHM

        return cls._get_moderator_width_analytical(moderator_data['type'],
                                                   moderator_data['parameters'],
                                                   moderator_data['scaling_function'],
                                                   moderator_data.get('scaling_parameters', None),
                                                   e_init)

    @staticmethod
    def _get_moderator_width_analytical(imod: int,
                                        mod_pars: list[float],
                                        scaling_function: str | None,
                                        scaling_parameters: None | list[float],
                                        e_init: float) -> float:
        """
        Analytically calculates the moderator contribution to the resolution squared, in FWHM.

        Parameters
        ----------
        imod
            The moderator type. See `Moderator.type`.
        mod_pars
            Moderator parameters. See `Moderator.parameters`.
        scaling_function
            The name of the scaling function to use. If None, no scaling will be applied. Only
            applicable for type 2 moderators (modified chi squared).
        scaling_parameters
            The parameters for the `scaling_function`. Only required when `scaling_function` is
            provided.
        e_init
            The initial energy, as selected by the user.

        Returns
        -------
        moderator_width_squared
            The moderator contribution to the resolution squared, in FWHM.

        Raises
        ------
        NotImplementedError
            If the moderator type is not one of the defined values.
        """
        # TODO: Look into composition
        if imod == 0:
            return np.array(mod_pars) * 1e-3 / 1.95 / (437.392 * np.sqrt(e_init)) ** 2 * SIGMA2FWHMSQ
        elif imod == 1:
            return PyChopModel._get_moderator_width_ikeda_carpenter(*mod_pars, e_init=e_init)
        elif imod == 2:
            ei_sqrt = np.sqrt(e_init)
            delta_0, delta_G = mod_pars[0] * 1e-3, mod_pars[1] * 1e-3

            if scaling_function is not None:
                func = MODERATOR_MODIFICATION_FUNCTIONS[scaling_function]
                delta_0 *= func(e_init, scaling_parameters)

            return ((delta_0 + delta_G * ei_sqrt) / 1.96 / (437.392 * ei_sqrt)) ** 2 * SIGMA2FWHMSQ
        elif imod == 3:
            return Polynomial(mod_pars)(np.sqrt(E2L / e_init)) ** 2 * 1e-12
        else:
            raise NotImplementedError()

    @staticmethod
    def _get_moderator_width_ikeda_carpenter(s1: float,
                                             s2: float,
                                             b1: float,
                                             b2: float,
                                             e_mod: float,
                                             e_init: float) -> float:
        """
        Calculates the moderator time width based on the Ikeda-Carpenter distribution.

        Parameters
        ----------
        s1
        s2
        b1
        b2
        e_mod
        e_init
            The initial energy, as selected by the user.

        Returns
        -------
        moderator_width_squared
            The moderator contribution to the resolution squared, in FWHM.
        """
        sig = np.sqrt(s1 ** 2 + s2 ** 2 * 81.8048 / e_init)
        a = 4.37392e-4 * sig * np.sqrt(e_init)
        b = b2 if e_init > 130. else b1
        r = np.exp(- e_init / e_mod)
        return (3. / a ** 2 + (r * (2. - r)) / b ** 2) * 1e-12 * SIGMA2FWHMSQ

    @classmethod
    @abstractmethod
    def _get_chopper_width_squared(cls,
                                   model_data: PyChopModelData,
                                   e_init: float,
                                   chopper_frequency: list[int]) -> tuple[float, float | None]:
        raise NotImplementedError()

    @staticmethod
    def _get_distances(choppers: dict[str, FermiChopper | DiskChopper]
                       ) -> tuple[float, float, float]:
        """
        Determines various distances in the instrument.

        The choppers in `PyChopModelData` must be present in the order of increasing distance from
        the moderator.

        Parameters
        ----------
        choppers
            Data for all choppers in the instrument.

        Returns
        -------
        d_moderator_last_chopper
            Distance between the moderator and the last chopper.
        d_aperture_last_chopper
            Distance between the aperture and the last chopper.
        d_moderator_first_chopper
            Distance between the moderator and the first chopper.
        """
        choppers: list[FermiChopper | DiskChopper] = list(choppers.values())
        mod_chop = choppers[-1]['distance']
        try:
            ap_chop = choppers[-1]['aperture_distance']
        except KeyError:
            ap_chop = mod_chop

        return mod_chop, ap_chop, choppers[0]['distance']

    @classmethod
    def _get_detector_width_squared(cls,
                                    detector_data: Detector,
                                    fake_frequencies: Float[np.ndarray, 'frequency'],
                                    e_init: float) -> Float[np.ndarray, 'detector_width_squared']:
        """
        Calculates the detector contribution to the resolution squared, in FWHM.

        Parameters
        ----------
        detector_data
            The data for the detector.
        fake_frequencies
            The coarse grid of energy transfer at which to evaluate the contribution to resolution.
        e_init
            The initial energy, as selected by the user.

        Returns
        -------
        detector_width_squared
            The detector contribution to the resolution squared, in FWHM.

        Raises
        ------
        NotImplementedError
            If the detector type is 1 (Lithium detector).

        See Also
        --------
        _get_he_detector_width_squared : Computes the detector contribution for Helium detectors
        """
        wfs = np.sqrt(E2K * (e_init - fake_frequencies))
        t2rad = 0.063
        atms = 10.
        const = 50.04685368

        if detector_data['type'] == 1:
            raise NotImplementedError()
        else:
            rad = detector_data['depth'] * 0.5
            reff = rad * (1.0 - t2rad)
            var = 2.0 * (rad * (1.0 - t2rad)) * (const * atms)
            alf = var / wfs

            assert not np.any(alf < 0.)

            return cls._get_he_detector_width_squared(alf) * reff ** 2 * SIGMA2FWHMSQ

    @staticmethod
    def _get_he_detector_width_squared(alf: Float[np.ndarray, 'ALF']) -> Float[np.ndarray, 'VX']:
        """
        Calculates the helium detector contribution to the resolution squared, in FWHM.

        T.G.Perring 6/4/90

        This method approximates the contribution using Chebyshev polynomial expansions over the
        ranges 0 <= `alf` <= 9 and 10 <= `alf`. For the intervening interval, 9 <= `alf` <= 10,
        a linear approximation of the two approximations is taken.

        The original Chebyshev coefficients were obtained using the ``CHEBFT`` routine from
        "numerical recipes", but the numpy equivalent for the ``CHEBEV``,
        `numpy.polynomial.chebyshev.Chebyshev`, uses a slightly different formalism, so the first
        coefficient is halved compared to the source implementation of this method.

        Parameters
        ----------
        alf
            ALF = radius in m.f.p.

        Returns
        -------
        vx
            Variance of depth absorption.
        """
        out = np.zeros(np.shape(alf))
        coefficients_low = [0.613452291529095, -0.3621914072547197, 6.0117947617747081e-02,
                  1.8037337764424607e-02, -1.4439005957980123e-02, 3.8147446724517908e-03, 1.3679160269450818e-05,
                  -3.7851338401354573e-04, 1.3568342238781006e-04, -1.3336183765173537e-05, -7.5468390663036011e-06,
                  3.7919580869305580e-06, -6.4560788919254541e-07, -1.0509789897250599e-07, 9.0282233408123247e-08,
                  -2.1598200223849062e-08, -2.6200750125049410e-10, 1.8693270043002030e-09, -6.0097600840247623e-10,
                  4.7263196689684150e-11, 3.3052446335446462e-11, -1.4738090470256537e-11, 2.1945176231774610e-12,
                  4.7409048908875206e-13, -3.3502478569147342e-13]

        coefficients_high = [0.9313232069059375, 7.5988886169808666e-02, -8.3110620384910993e-03,
                  1.1236935254690805e-03, -1.0549380723194779e-04, -3.8256672783453238e-05, 2.2883355513325654e-05,
                  -2.4595515448511130e-06, -2.2063956882489855e-06, 7.2331970290773207e-07, 2.2080170614557915e-07,
                  -1.2957057474505262e-07, -2.9737380539129887e-08, 2.2171316129693253e-08, 5.9127004825576534e-09,
                  -3.7179338302495424e-09, -1.4794271269158443e-09, 5.5412448241032308e-10, 3.8726354734119894e-10,
                  -4.6562413924533530e-11, -9.2734525614091013e-11, -1.1246343578630302e-11, 1.6909724176450425e-11,
                  5.6146245985821963e-12, -2.7408274955176282e-12]

        g0 = (32.0 - 3.0 * (np.pi ** 2)) / 48.0
        g1 = 14.0 / 3.0 - (np.pi ** 2) / 8.0

        chebyshev_low = np.polynomial.Chebyshev(coefficients_low, [0., 10.])
        chebyshev_high = np.polynomial.Chebyshev(coefficients_high, [-1., 1.])

        first_indices = alf <= 9.
        last_indices = alf >= 10.
        mid_indices = np.logical_not(np.logical_or(first_indices, last_indices))

        out[first_indices] = 0.25 * chebyshev_low(alf[first_indices])
        out[last_indices] = g0 + g1 * chebyshev_high(1. - 18. / alf[last_indices]) / alf[last_indices] ** 2

        mid_alf = alf[mid_indices]
        guess1 = 0.25 * chebyshev_low(mid_alf)
        guess2 = g0 + g1 * chebyshev_high(1. - 18. / mid_alf) / mid_alf ** 2
        out[mid_indices] = (10. - mid_alf) * guess1 + (mid_alf - 9.) * guess2

        return out

    @staticmethod
    def _get_sample_width_squared(sample_data: Sample) -> float:
        """
        Calculates the sample contribution to the resolution squared, in FWHM.

        Parameters
        ----------
        sample_data
            The data for the sample.

        Returns
        -------
        sample_width_squared
            The sample contribution to the resolution squared, in FWHM.
        """
        scaling_factor = 0.125 if sample_data['type'] == 2 else 1 / 12
        return sample_data['width'] ** 2 * scaling_factor * SIGMA2FWHMSQ

    # def _create_tau_resolution(self, model: str,
    #                           setting: list[str],
    #                           e_init: float,
    #                           chopper_frequency: float
    #                            ) -> Callable[[Float[np.ndarray, 'frequencies']], Float[np.ndarray, 'sigma']]:
    #     model_data = self.models[model]['parameters']
    #
    #     tsq_moderator = self.get_moderator_width(params['measured_width'], e_init, params['imod']) ** 2
    #     tsq_chopper = self.get_chopper_width_squared(setting, True, e_init, chopper_frequency)
    #
    #     l0 = self.constants['Fermi']['distance']
    #     l1 = self.constants['d_chopper_sample']
    #     l2 = self.constants['d_sample_detector']
    #
    #     def resolution(frequencies: Float[np.ndarray, 'frequencies']) -> Float[np.ndarray, 'sigma']:
    #         e_final = frequencies - e_init
    #         energy_term = (e_final / e_init) ** 1.5
    #
    #         term1 = (1 + (l0 + l1) / l2 * energy_term) ** 2
    #         term2 = (1 + l1 / l2 * energy_term) ** 2
    #
    #     return resolution


class PyChopModelFermi(PyChopModel):
    """
    PyChop :term:`model` of 2D direct-geometry INS :term:`instruments<instrument>` that use a
    :term:`Fermi chopper`.

    Models the :term:`resolution` as a function of energy transfer (frequencies) only, with the
    output :term:`model` being a Gaussian. This is done by computing the contribution of each part
    of the :term:`instrument` to the :term:`resolution function`. However, this model calculates the
    :term:`resolution` on a coarse grid, and then fits a polynomial to the results - the
    :term:`resolution` at the user-provided ``frequencies`` is obtained by evaluating the polynomial.

    Parameters
    ----------
    model_data
        The data associated with the model for a given version of a given instrument.
    e_init
        The initial energy in meV used in the INS experiment. If not provided, the default value for
        the particular version of an instrument will be used
        (see `PyChopModelDataFermi.defaults`). Please note that the `e_init` value must be
        within the range allowed for the instrument (see `PyChopModelDataFermi.restrictions`).
    chopper_frequency
        The frequency of the Fermi chopper in Hz used in the INS experiment. If not provided, the
        default value for the particular version of an instrument will be used
        (see `PyChopModelDataFermi.defaults`). Please note that the `chopper_frequency` value must
        be within the range allowed for the instrument (see `PyChopModelDataFermi.restrictions`).
    fitting_order
        The order of the polynomial to use when fitting to the coarse grid.

    Raises
    ------
    InvalidInputError
        If either the `e_init` or the `chopper_frequency` is not allowed.
    NoTransmissionError
        If the combination of `e_init` and `chopper_frequency` would result in the Fermi chopper
        blocking the neutron beam, resulting in no signal.

    Attributes
    ----------
    input
        The input that the ``__call__`` method expects.
    output
        The output of the ``__call__`` method.
    data_class
        Reference to the `PyChopModelDataFermi` type.
    citation
    polynomial
    """
    data_class = PyChopModelDataFermi

    def __init__(self,
                 model_data: PyChopModelDataFermi,
                 e_init: Optional[float] = None,
                 chopper_frequency: Optional[int] = None,
                 fitting_order: int = 4,
                 **_):
        super().__init__(model_data)
        
        if chopper_frequency is None:
            chopper_frequency = model_data.default_chopper_frequency
        elif chopper_frequency not in range(*model_data.allowed_chopper_frequencies):
            raise InvalidInputError(f'The provided chopper frequency ({chopper_frequency}) is not '
                                    f'allowed; only the following frequencies are possible: '
                                    f'{list(range(*model_data.allowed_chopper_frequencies))}')

        e_init = self._validate_e_init(e_init, model_data)

        fake_frequencies, resolution = self._precompute_resolution(model_data, e_init, [chopper_frequency])
        self._polynomial = Polynomial.fit(fake_frequencies, resolution, fitting_order)

    @property
    def polynomial(self) -> Polynomial:
        return self._polynomial

    @classmethod
    def _get_chopper_width_squared(cls,
                                   model_data: PyChopModelDataFermi,
                                   e_init: float,
                                   chopper_frequency: list[int]) -> tuple[float, None]:
        """
        Calculates the Fermi chopper contribution to the resolution squared, in FWHM.

        Parameters
        ----------
        model_data
            The data for a particular version of a particular instrument.
        e_init
            The initial energy, as selected by the user.
        chopper_frequency
            The frequency of the Fermi chopper, as selected by the user.

        Returns
        -------
        chopper_width_squared
            The chopper contribution to the resolution squared, in FWHM.
        none
            For compatibility with `PyChopModelNonFermi`.

        Raises
        ------
        NoTransmissionError
            If the combination of `e_init` and `chopper_frequency` would result in the Fermi chopper
            blocking the neutron beam, resulting in no signal.
        """
        frequency = 2 * np.pi * chopper_frequency[0]
        gamm = (2.00 * model_data.radius ** 2 / model_data.pslit) * \
               abs(1.00 / model_data.rho - 2.00 * frequency / (437.392 * np.sqrt(e_init)))

        if gamm >= 4.:
            raise NoTransmissionError(f'The combination of e_init={e_init} and chopper_frequency='
                                      f'{chopper_frequency} is not valid because the Fermi chopper '
                                      f'has no transmission at these values.')
        elif gamm <= 1.:
            gsqr = (1.00 - (gamm ** 2) ** 2 / 10.00) / (1.00 - (gamm ** 2) / 6.00)
        else:
            groot = np.sqrt(gamm)
            gsqr = 0.60 * gamm * ((groot - 2.00) ** 2) * (groot + 8.00) / (groot + 4.00)

        sigma = ((model_data.pslit / (2.00 * model_data.radius * frequency)) ** 2 / 6.00) * gsqr
        return sigma * SIGMA2FWHMSQ, None


class PyChopModelNonFermi(PyChopModel, ABC):
    """
    Abstract base class for PyChop models for instruments that do not have a Fermi chopper.

    This class contains methods for calculating the chopper contribution to the resolution function
    for all instruments that do not have a Fermi chopper, but it does not implement the abstract
    methods of its superclasses. Therefore, any of its subclasses *must implement* the ``__init__``
    method as well as the ``polynomial`` attribute.

    For most subclasses, the actual content of the ``__init__`` method should be nearly identical to
    that of the already existing subclasses; the only variable thing should be its signature.
    Different instruments have different sets of choppers, and so different sets of chopper
    frequencies have to be provided as a user input (mirroring the choice of the chopper frequencies
    on the physical INS instrument). Each subclass should have a separate argument for each of the
    user-tunable chopper frequencies, but then bundle them together to pass in to the
    `_validate_chopper_frequency` method which has an implementation in this class.
    """
    data_class = PyChopModelDataNonFermi

    @staticmethod
    def _validate_chopper_frequency(chopper_frequencies: list[int | None],
                                    model_data: PyChopModelDataNonFermi) -> list[int]:
        """
        Validates that the user-provided `chopper_frequencies` are among the allowed values for the instrument.

        Parameters
        ----------
        chopper_frequencies
            A list of chopper frequencies corresponding to the choppers with tunable frequencies.
            The list must have the same length and order as the
            `PyChopModelDataNonFermi.allowed_chopper_frequencies` for that instrument. Each entry
            which contains a ``None`` instead of a value will be replaced with the default value
            for the corresponding chopper.
        model_data
            The data for a particular INS instrument.

        Returns
        -------
        chopper_frequencies
            The valid chopper frequencies.

        Raises
        ------
        InvalidInputError
            If any of the provided `chopper_frequencies` is invalid.
        """
        for i, (frequency, allowed_chopper_frequencies) in (
                enumerate(zip(chopper_frequencies, model_data.allowed_chopper_frequencies))):
            if frequency is None:
                chopper_frequencies[i] = model_data.default_chopper_frequency[i]
            elif frequency not in range(*allowed_chopper_frequencies):
                raise InvalidInputError(
                    f'The provided chopper frequency ({frequency}) is not allowed; only the'
                    f' following frequencies are possible: '
                    f'{list(range(*allowed_chopper_frequencies))}')

        return chopper_frequencies

    @staticmethod
    def get_long_frequency(frequencies: list[int],
                           model_data: PyChopModelDataNonFermi
                           ) -> Float[np.ndarray, 'chopper_frequencies']:
        """
        Calculates the frequency of each chopper making up this instrument.

        Different instruments are set up differently; usually, at least one of the choppers is
        controllable by the user, and others are run at constant frequencies. However, in some
        instruments, some of the choppers are run at a particular fraction of another chopper's
        frequency (see `PyChopModelDataNonFermi.frequency_matrix`). This function takes everything
        into account and computes the frequency of each chopper.

        Parameters
        ----------
        frequencies
            The frequency of each user-controllable chopper (in Hz), in the same order as in the
            ``__init__``.
        model_data
            The data for a particular INS instrument.

        Returns
        -------
        all_frequencies
            The frequency of each chopper, in the order of increasing distance from the moderator.
        """
        frequencies += model_data.default_chopper_frequency[len(frequencies):]
        frequency_matrix = np.array(model_data.frequency_matrix)

        return np.dot(frequency_matrix, frequencies) + model_data.constant_frequencies

    @classmethod
    def _get_chop_times(cls,
                        model_data: PyChopModelDataNonFermi,
                        e_init: float,
                        chopper_frequency: list[int]) -> list[list[Float[np.ndarray, 'times']]]:
        """
        Calculates the chop times of the first and last choppers.

        This information can be used to compute the chopper contribution to the resolution. If the
        instrument contains only one chopper, a fake chopper is prepended.

        Parameters
        ----------
        model_data
            The data for a particular INS instrument.
        e_init
            The initial energy, as selected by the user.
        chopper_frequency
            The frequency of all user-controlled choppers.

        Returns
        -------
        chop_times
            The chop times of the first and last chopper.
        """
        frequencies = cls.get_long_frequency(chopper_frequency, model_data)
        choppers = model_data.choppers

        # conversion factors
        lam2TOF = 252.7784  # the conversion from wavelength to TOF at 1m, multiply by distance
        uSec = 1e6  # seconds to microseconds
        lam = np.sqrt(81.8042 / e_init)  # convert from energy to wavelenth

        p_frames = model_data.source_frequency / model_data.n_frame

        # if there's only one disk we prepend a dummy disk with full opening at zero distance
        # so that the distance calculations (which needs the difference between disk pos) works
        if len(choppers) == 1:
            choppers = [
                {
                    'distance': 0,
                    'nslot': 1,
                    'slot_ang_pos': None,
                    'slot_width': 3141,
                    'guide_width': 10,
                    'radius': 500,
                    'num_disk': 1
                },
                *list(choppers.values())
            ]
            frequencies = np.array([model_data.source_frequency, frequencies[0]])
        else:
            choppers = list(choppers.values())

        chop_times = []

        # first we optimise on the main Ei
        for frequency, chopper in zip([frequencies[0], frequencies[-1]], [choppers[0], choppers[-1]]):
            chopper: DiskChopper
            this_phase, phase_independence = chopper['default_phase'], chopper['is_phase_independent']

            # checks whether this chopper should have an independently set phase / delay
            is_phase_str = isinstance(this_phase, str)
            islt = int(this_phase) if (phase_independence and is_phase_str) else 0

            if phase_independence and not is_phase_str:
                # effective chopper velocity (if 2 disks effective velocity is double)
                chopVel = 2 * np.pi * chopper['radius'] * chopper['num_disk'] * frequency

                # full opening time
                t_full_op = uSec * (chopper['slot_width'] + chopper['guide_width']) / chopVel
                realTimeOp = np.array([this_phase, this_phase + t_full_op])
            else:
                # the opening time of the chopper so that it is open for the focus wavelength
                t_open = lam2TOF * lam * chopper['distance']
                # effective chopper velocity (if 2 disks effective velocity is double)
                chopVel = 2 * np.pi * chopper['radius'] * chopper['num_disk'] * frequency

                # full opening time
                t_full_op = uSec * (chopper['slot_width'] + chopper['guide_width']) / chopVel
                # set the chopper phase to be as close to zero as possible
                realTimeOp = np.array([(t_open - t_full_op / 2.0), (t_open + t_full_op / 2.0)])

            slot, angles = chopper['nslot'], chopper['slot_ang_pos']

            chop_times.append([])
            if slot > 1 and angles:
                tslots = [(uSec * angles[j] / 360.0 / frequency) for j in range(slot)]
                tslots = [[(t + r * (uSec / frequency)) - tslots[0] for r in range(int(frequency / p_frames))] for t in
                          tslots]
                realTimeOp -= np.max(tslots[islt % slot])
                islt = 0
                next_win_t = uSec / model_data.source_frequency + (uSec / frequency)

                while realTimeOp[0] < next_win_t:
                    chop_times[-1].append(deepcopy(realTimeOp))
                    slt0 = islt % slot
                    slt1 = (islt + 1) % slot
                    angdiff = angles[slt1] - angles[slt0]
                    if (slt1 - slt0) != 1:
                        angdiff += 360
                    realTimeOp += uSec * (angdiff / 360.0) / frequency
                    islt += 1
            else:
                # If angular positions of slots not defined, assumed evenly spaced (LET, MERLIN)
                next_win_t = uSec / (slot * frequency)
                realTimeOp -= next_win_t * np.ceil(realTimeOp[0] / next_win_t)

                while realTimeOp[0] < (uSec / p_frames + next_win_t):
                    chop_times[-1].append(deepcopy(realTimeOp))
                    realTimeOp += next_win_t

        return chop_times

    @classmethod
    def _get_chopper_width_squared(cls,
                                   model_data: PyChopModelDataNonFermi,
                                   e_init: float,
                                   chopper_frequency: list[int]) -> tuple[float, float]:
        """
        Calculates the chopper contribution to the resolution squared, in FWHM.

        Parameters
        ----------
        model_data
            The data for a particular INS instrument.
        e_init
            The initial energy, as selected by the user.
        chopper_frequency
            The frequency of all user-controlled choppers.

        Returns
        -------
        last_chopper_width_squared
            The last chopper's contribution to the resolution squared, in FWHM.
        first_chopper_width_squared
            The first chopper's contribution to the resolution squared, in FWHM.
        """
        chop_times = cls._get_chop_times(model_data, e_init, chopper_frequency)

        wd0 = (chop_times[-1][0][1] - chop_times[-1][0][0]) * 0.5e-6
        wd1 = (chop_times[0][0][1] - chop_times[0][0][0]) * 0.5e-6

        return wd0 ** 2, wd1 ** 2


class PyChopModelCNCS(PyChopModelNonFermi):
    """
    A PyChop :term:`model` for the CNCS :term:`instrument`.

    This :term:`model` is identical to all other PyChop models for instruments without a
    :term:`Fermi chopper`, but the user-choice :term:`chopper` frequencies have unique names
    compared to the other models.

    Parameters
    ----------
    model_data
        The data for the PyChopModel of the CNCS instrument.
    e_init
        The incident energy used in the INS experiment. If not provided, the default value for
        the particular version of the CNCS instrument will be used
        (see `PyChopModelDataNonFermi.defaults`). Please note that the `e_init` value must be
        within the range allowed for the instrument (see `PyChopModelDataNonFermi.restrictions`).
    resolution_disk_frequency
        The frequency of the resolution disk chopper (chopper 4). If not provided, the
        default value for the particular version of the CNCS instrument will be used (see
        `PyChopModelDataNonFermi.defaults`). Please note that the `resolution_disk_frequency` value must
        be within the range allowed for the instrument (see `PyChopModelDataNonFermi.restrictions`).
    fermi_frequency
        The frequency of the Fermi chopper (chopper 1). If not provided, the
        default value for the particular version of the CNCS instrument will be used
        (see `PyChopModelDataNonFermi.defaults`). Please note that the `fermi_frequency` value must be
        within the range allowed for the instrument (see `PyChopModelDataNonFermi.restrictions`).
    fitting_order
        The order of the polynomial used for fitting against the resolution.

    Raises
    ------
    InvalidInputError
        If any of `e_init`, `resolution_disk_frequency`, or `fermi_frequency` is not allowed.

    Attributes
    ----------
    input
        The input that the ``__call__`` method expects.
    output
        The output of the ``__call__`` method.
    data_class
        Reference to the `PyChopModelDataNonFermi` type.
    citation
    polynomial
    """
    def __init__(self,
                 model_data: PyChopModelDataNonFermi,
                 e_init: Optional[float] = None,
                 resolution_disk_frequency: Optional[int] = None,
                 fermi_frequency: Optional[int] = None,
                 fitting_order: int = 4,
                 **_):
        super().__init__(model_data)

        chopper_frequencies = [resolution_disk_frequency, fermi_frequency]
        chopper_frequencies = self._validate_chopper_frequency(chopper_frequencies, model_data)

        e_init = self._validate_e_init(e_init, model_data)

        frequencies, resolution = self._precompute_resolution(model_data, e_init, chopper_frequencies)
        self._polynomial = Polynomial.fit(frequencies, resolution, fitting_order)

    @property
    def polynomial(self):
        return self._polynomial


class PyChopModelLET(PyChopModelNonFermi):
    """
    A PyChop :term:`model` for the LET :term:`instrument`.

    This :term:`model` is identical to all other PyChop models for instruments without a
    :term:`Fermi chopper`, but the user-choice :term:`chopper` frequencies have unique names
    compared to the other models.

    The LET instrument, specifically, has a set-up with multiple choppers of variable frequency, but
    where some of the choppers are set to a pre-determined fraction of the frequency of another
    chopper. Further, this relationship changes depending on the ``chopper_package`` setting. The
    `PyChopModelDataNonFermi.frequency_matrix` attribute describes this relationship, and the
    `get_long_frequency` method can be used to compute the frequencies of all choppers.

    Parameters
    ----------
    model_data
        The data for the PyChopModel of the LET instrument.
    e_init
        The incident energy used in the INS experiment. If not provided, the default value for
        the particular version of the LET instrument will be used
        (see `PyChopModelDataNonFermi.defaults`). Please note that the `e_init` value must be
        within the range allowed for the instrument (see `PyChopModelDataNonFermi.restrictions`).
    resolution_frequency
        The frequency of the resolution chopper (i.e. the second resolution disk chopper, or chopper
        5). If not provided, the default value for the particular version of the LET instrument will
        be used (see `PyChopModelDataNonFermi.defaults`). Please note that the
        `resolution_frequency` value must be within the range allowed for the instrument
        (see `PyChopModelDataNonFermi.restrictions`).
    pulse_remover_frequency
        The frequency of the pulse remover disk chopper (chopper 3). If not provided, the
        default value for the particular version of the LET instrument will be used (see
        `PyChopModelDataNonFermi.defaults`). Please note that the `pulse_remover_frequency` value
        must be within the range allowed for the instrument (see
        `PyChopModelDataNonFermi.restrictions`).
    fitting_order
        The order of the polynomial used for fitting against the resolution.

    Raises
    ------
    InvalidInputError
        If any of `e_init`, `resolution_frequency`, or `pulse_remover_frequency` is not allowed.

    Attributes
    ----------
    input
        The input that the ``__call__`` method expects.
    output
        The output of the ``__call__`` method.
    data_class
        Reference to the `PyChopModelDataNonFermi` type.
    citation
    polynomial
    """
    def __init__(self,
                 model_data: PyChopModelDataNonFermi,
                 e_init: Optional[float] = None,
                 resolution_frequency: Optional[int] = None,
                 pulse_remover_frequency: Optional[int] = None,
                 fitting_order: int = 4,
                 **_):
        super().__init__(model_data)

        chopper_frequencies = [resolution_frequency, pulse_remover_frequency]
        chopper_frequencies = self._validate_chopper_frequency(chopper_frequencies, model_data)

        e_init = self._validate_e_init(e_init, model_data)

        frequencies, resolution = self._precompute_resolution(model_data, e_init, chopper_frequencies)
        self._polynomial = Polynomial.fit(frequencies, resolution, fitting_order)

    @property
    def polynomial(self):
        return self._polynomial


def soft_hat(x: float, p: list[float]):
    """
    Soft hat function, from Herbert subroutine library.

    Used for some versions of some instruments for rescaling t-mod at low energy to account for
    broader moderator term.

    Parameters
    ----------
    x
        The inital energy in meV, as provided by the user.
    p
        A list of parameters to use for the scaling.

    Returns
    -------
    y
        The scaling factor
    """
    x = np.array(x)
    sig2fwhh = np.sqrt(8 * np.log(2))
    height, grad, x1, x2 = tuple(p[:4])
    sig1, sig2 = abs(p[4] / sig2fwhh), abs(p[5] / sig2fwhh)
    # linearly interpolate sig for x1<x<x2
    sig = ((x2 - x) * sig1 - (x1 - x) * sig2) / (x2 - x1)
    if np.shape(sig):
        sig[x < x1] = sig1
        sig[x > x2] = sig2
    # calculate blurred hat function with gradient
    e1 = (x1 - x) / (np.sqrt(2) * sig)
    e2 = (x2 - x) / (np.sqrt(2) * sig)
    y = (erf(e2) - erf(e1)) * ((height + grad * (x - (x2 + x1) / 2)) / 2)
    return y + 1


MODERATOR_MODIFICATION_FUNCTIONS = {
    'soft_hat': soft_hat
}


class NoTransmissionError(Exception):
    """
    A custom Exception used in `PyChopModelFermi` to signal that no transmission can occur.

    See `PyChopModelFermi._get_chopper_width_squared`.
    """
    pass
