"""Validate PyChop_fit resolutions against reference PyChop library"""

from __future__ import annotations

import itertools
import random

from more_itertools import sample as reservoir_sample
import numpy as np
from numpy.testing import assert_allclose
import pytest

from PyChop.Instruments import Instrument as PyChopInstrument
from PyChop.Chop import tube_mts
from PyChop.MulpyRep import calcChopTimes

from resolution_functions.instrument import Instrument
from resolution_functions.models.pychop import *
from resolution_functions.models.model_base import InvalidInputError, ModelData

if TYPE_CHECKING:
    from jaxtyping import Float


random.seed(1)

DEBUG = False
N_SAMPLES = 10

EINIT = np.arange(50, 2000, 50)
EINIT_SAMPLE = reservoir_sample(EINIT, k=N_SAMPLES)

CHOPPER_FREQ_FERMI = np.arange(50, 601, 50)
MATRIX_FERMI = list(reservoir_sample(
    itertools.product(EINIT, CHOPPER_FREQ_FERMI),
    k=N_SAMPLES)
)

CHOPPER_FREQ_NONFERMI = np.arange(60, 301, 60)
MATRIX_NONFERMI = list(reservoir_sample(
    itertools.product(EINIT, CHOPPER_FREQ_NONFERMI, CHOPPER_FREQ_NONFERMI),
    k=N_SAMPLES)
)

def matrix_fermi_id(matrix_row: tuple[int, int]) -> str:
    """Get test id string from Fermi chopper frequencies"""
    e_init, frequency = matrix_row
    return f"e_init={e_init},f={frequency}"


def matrix_nonfermi_id(matrix_row: tuple[int, int, int]) -> str:
    """Get test id string from Non-Fermi chopper frequencies"""
    e_init, f1, f2 = matrix_row
    return f"e_init={e_init},f1={f1},f2={f2}"


INSTRUMENTS_FERMI = [
    [('ARCS', 'ARCS')],
    [('HYSPEC', 'HYSPEC')],
    [('MAPS', 'MAPS')],
    [('MARI', 'MARI')],
    [('MERLIN', 'MERLIN')],
    [('SEQUOIA', 'SEQUOIA')],
]

INSTRUMENT_SETTINGS_FERMI = [
    ['SEQ-100-2.0-AST', 'SEQ-700-3.5-AST', 'ARCS-100-1.5-AST', 'ARCS-700-1.5-AST',
     'ARCS-700-0.5-AST', 'ARCS-100-1.5-SMI', 'ARCS-700-1.5-SMI'],
    ['OnlyOne'],
    ['A', 'S'],
    ['A', 'B', 'C', 'G', 'R', 'S'],
    ['G', 'S'],
    ['Fine', 'Sloppy', 'SEQ-100-2.0-AST', 'SEQ-700-3.5-AST', 'ARCS-100-1.5-AST', 'ARCS-700-1.5-AST',
     'ARCS-700-0.5-AST', 'ARCS-100-1.5-SMI', 'ARCS-700-1.5-SMI'],
]

INSTRUMENT_MATRIX_FERMI = list(
        itertools.chain.from_iterable(
            itertools.product(instr, settings)
                  for instr, settings in zip(INSTRUMENTS_FERMI, INSTRUMENT_SETTINGS_FERMI)
        )
)

INSTRUMENTS_NONFERMI = [
    [('CNCS', 'CNCS')],
    [('LET', 'LET')],
]
INSTRUMENT_SETTINGS_NONFERMI = [
    ['High Flux', 'Intermediate', 'High Resolution'],
    ['High Flux', 'Intermediate', 'High Resolution'],
]
INSTRUMENT_MATRIX_NONFERMI = list(
        itertools.chain.from_iterable(
            itertools.product(instr, settings)
                  for instr, settings in zip(INSTRUMENTS_NONFERMI, INSTRUMENT_SETTINGS_NONFERMI)
        )
)


def instrument_id(matrix_row: tuple[tuple[str, str], str]) -> str:
    """Get test id NAME_SETTING from input row ((NAME, NAME), SETTING)"""
    (instrument, _), setting = matrix_row
    return f"{instrument}_{setting}"

# id formatter for E_i input
format_ei = "ei={}".format


def get_fake_frequencies(e_init: float) -> Float[np.ndarray]:
    return np.linspace(0, e_init, 40, endpoint=False)


@pytest.fixture(scope="module", params=INSTRUMENT_MATRIX_FERMI, ids=instrument_id)
def pychop_fermi_data(request) -> tuple[PyChopModelDataFermi, PyChopInstrument]:
    (name, version), setting = request.param
    maps = Instrument.from_default(name, version)
    rf = maps.get_model_data('PyChop_fit', chopper_package=setting)

    pc = PyChopInstrument(name, chopper=setting)
    return rf, pc


@pytest.fixture(scope="module", params=INSTRUMENT_MATRIX_NONFERMI, ids=instrument_id)
def pychop_nonfermi_data(request) -> tuple[PyChopModelDataNonFermi, PyChopInstrument]:
    (name, version), setting = request.param
    maps = Instrument.from_default(name, version)
    rf = maps.get_model_data('PyChop_fit', chopper_package=setting)

    pc = PyChopInstrument(name, chopper=setting)
    return rf, pc


@pytest.fixture(scope="module")
def mari_data():
    maps = Instrument.from_default('MARI', 'MARI')
    rf = maps.get_model_data('PyChop_fit')

    pc = PyChopInstrument('MARI', chopper=maps.default_option_for_setting('PyChop_fit', 'chopper_package'))
    return rf, pc


@pytest.fixture(scope="module")
def cncs_data():
    cncs = Instrument.from_default('CNCS', 'CNCS')
    rf = cncs.get_model_data('PyChop_fit')
    return rf


@pytest.fixture(scope="module")
def let_data():
    cncs = Instrument.from_default('LET', 'LET')
    rf = cncs.get_model_data('PyChop_fit')
    return rf


@pytest.mark.parametrize(
    "chopper_frequency",
    [
        49.99999,
        -0.048,
        -np.inf,
        600.00017,
        np.inf,
        13554,
        np.nan,
        50.5,
        57.5,
        500.0000001,
        480,
    ],
)
def test_fermi_invalid_chopper_frequency(
    chopper_frequency, mari_data: tuple[PyChopModelDataFermi, PyChopInstrument]
):
    with pytest.raises(InvalidInputError, match="The provided chopper frequency"):
        PyChopModelFermi(mari_data[0], chopper_frequency=chopper_frequency)


@pytest.mark.parametrize(
    "e_init", [-5, -0.00048, -np.inf, 2000.1, np.inf, 13554.1654, np.nan]
)
def test_fermi_invalid_e_init(
    e_init, mari_data: tuple[PyChopModelDataFermi, PyChopInstrument]
):
    with pytest.raises(InvalidInputError, match="The provided incident energy"):
        PyChopModelFermi(mari_data[0], e_init=e_init)

@pytest.mark.parametrize(
    "chopper_frequency",
    [
        [59.99999, 60],
        [-0.048] * 2,
        [-np.inf, 0],
        [120, 300.00017],
        [np.inf, np.inf],
        [300, np.nan],
        [60.5, 60],
        [180, 67.5],
        [600, 600],
        [130, 130],
    ],
)
def test_cncs_invalid_chopper_frequency(chopper_frequency, cncs_data: PyChopModelDataNonFermi):
    with pytest.raises(InvalidInputError, match="The provided chopper frequency"):
        PyChopModelCNCS(cncs_data, resolution_disk_frequency=chopper_frequency[0], fermi_frequency=chopper_frequency[1])


@pytest.mark.parametrize('e_init', [-5, -0.00048, -np.inf, 80.1, np.inf, 13554.1654, np.nan])
def test_cncs_invalid_e_init(e_init, cncs_data: PyChopModelDataNonFermi):
    with pytest.raises(InvalidInputError, match="The provided incident energy"):
        PyChopModelCNCS(cncs_data, e_init=e_init)


@pytest.mark.parametrize(
    'chopper_frequency',
    [[59.99999, 60],
     [-0.048] * 2,
     [-np.inf, 0],
     [120, 300.00017],
     [np.inf, np.inf],
     [300, np.nan],
     [60.5, 60],
     [180, 67.5],
     [600, 600],
     [135, 135]]
)
def test_let_invalid_chopper_frequency(chopper_frequency, let_data: PyChopModelDataNonFermi):
    with pytest.raises(InvalidInputError, match="The provided chopper frequency"):
        PyChopModelLET(let_data, resolution_frequency=chopper_frequency[0], pulse_remover_frequency=chopper_frequency[1])


@pytest.mark.parametrize('e_init', [-5, -0.00048, -np.inf, 30.0001, np.inf, 13554.1654, np.nan])
def test_let_invalid_e_init(e_init, let_data: PyChopModelDataNonFermi):
    with pytest.raises(InvalidInputError, match="The provided incident energy"):
        PyChopModelLET(let_data, e_init=e_init)


def test_distances(mari_data: tuple[PyChopModelData, PyChopInstrument]):
    maps_data, pychop = mari_data
    x0, xa, xm = PyChopModelFermi._get_distances(maps_data.choppers)
    expected = pychop.chopper_system.getDistances()

    assert x0 == expected[0]
    assert xa == expected[1]
    assert xm == expected[-1]


@pytest.mark.parametrize('e_init', EINIT_SAMPLE, ids=format_ei)
def test_fermi_moderator_width_analytical(e_init, pychop_fermi_data):
    _test_moderator_width_analytical(e_init, *pychop_fermi_data, PyChopModelFermi)


@pytest.mark.parametrize('e_init', EINIT_SAMPLE, ids=format_ei)
def test_nonfermi_moderator_width_analytical(e_init, pychop_nonfermi_data):
    _test_moderator_width_analytical(e_init, *pychop_nonfermi_data, PyChopModelNonFermi)


def _test_moderator_width_analytical(e_init, data, pychop, cls):
    data = data.moderator
    actual = cls._get_moderator_width_analytical(data['type'], data['parameters'], data['scaling_function'],
                                                 data.get('scaling_parameters'), e_init)
    expected = pychop.moderator.getAnalyticWidthsSquared(e_init)

    assert_allclose(actual, expected)


@pytest.mark.parametrize('e_init', EINIT_SAMPLE, ids=format_ei)
def test_fermi_moderator_width(e_init, pychop_fermi_data):
    _test_moderator_width(e_init, PyChopModelFermi, *pychop_fermi_data)


@pytest.mark.parametrize('e_init', EINIT_SAMPLE, ids=format_ei)
def test_nonfermi_moderator_width(e_init, pychop_nonfermi_data):
    _test_moderator_width(e_init, PyChopModelNonFermi, *pychop_nonfermi_data)


def _test_moderator_width(e_init, cls, data, pychop):
    actual = cls._get_moderator_width_squared(data.moderator, e_init)
    expected = pychop.moderator.getWidthSquared(e_init)

    assert_allclose(actual, expected)


@pytest.mark.parametrize('matrix', MATRIX_FERMI, ids=matrix_fermi_id)
def test_fermi_chopper_width(matrix, pychop_fermi_data):
    e_init, chopper_frequency = matrix
    pychop_fermi_data, pychop = pychop_fermi_data

    try:
        pychop.chopper_system.setFrequency(chopper_frequency)
    except ValueError as e:
        if 'maximum allowed' in str(e):
            pytest.skip('Frequency outside the bounds of this instrument')
    expected = pychop.chopper_system.getWidthSquared(e_init)

    if np.isnan(expected[0]):
        with pytest.raises(NoTransmissionError):
            PyChopModelFermi._get_chopper_width_squared(pychop_fermi_data, e_init, [chopper_frequency])
        return

    actual = PyChopModelFermi._get_chopper_width_squared(pychop_fermi_data, e_init, [chopper_frequency])

    assert_allclose(actual[0], expected[0], rtol=0, atol=1e-8)
    assert actual[1] is None


@pytest.mark.parametrize('matrix', MATRIX_NONFERMI, ids=matrix_nonfermi_id)
def test_nonfermi_chopper_width(matrix, pychop_nonfermi_data):
    e_init, *chopper_frequencies = matrix
    data, pychop = pychop_nonfermi_data

    pychop.chopper_system.setFrequency(chopper_frequencies)
    expected = pychop.chopper_system.getWidthSquared(e_init)

    actual = PyChopModelNonFermi._get_chopper_width_squared(data, e_init, chopper_frequencies)

    assert_allclose(actual[0], expected[0], rtol=0, atol=1e-8)
    assert_allclose(actual[1], expected[1], rtol=0, atol=1e-8)


@pytest.mark.parametrize('matrix', MATRIX_NONFERMI, ids=matrix_nonfermi_id)
def test_long_frequency(matrix, pychop_nonfermi_data):
    e_init, *chopper_frequencies = matrix
    data, pychop = pychop_nonfermi_data

    pychop.chopper_system.setFrequency(chopper_frequencies)
    expected = pychop.chopper_system._long_frequency

    actual = PyChopModelNonFermi.get_long_frequency(chopper_frequencies, data)

    assert_allclose(actual, expected)


@pytest.mark.parametrize('matrix', MATRIX_NONFERMI, ids=matrix_nonfermi_id)
def test_chop_times(matrix, pychop_nonfermi_data):
    e_init, *chopper_frequencies = matrix
    data, pychop = pychop_nonfermi_data

    pychop.chopper_system.setFrequency(chopper_frequencies)
    _, expected, _, _, _ = calcChopTimes(e_init, pychop.chopper_system._long_frequency, pychop.chopper_system._instpar,
                                  pychop.chopper_system.phase)

    actual = PyChopModelNonFermi._get_chop_times(data, e_init, chopper_frequencies)

    for aa, ee in zip(actual, [expected[0], expected[-1]]):
        for a, e in zip(aa, ee):
            try:
                assert_allclose(a, e, rtol=0, atol=1e-8)
            except AssertionError:
                print(actual)
                print(expected)
                raise


def test_he_detector_width_squared():
    data = np.array([7.19448838, 11.06500966, 12.43517103, 3.9478169, 10.27173622,
                     2.02389188, 3.42669603, 16.25052101, 14.55653963, 10.65643823,
                     1.47904548, 0.16923375, 10.19567838, 13.56701051, 9.96840209,
                     5.01442892, 1.53962029, 12.4874124, 19.08556836, 0.23525492,
                     3.51331984, 13.66939967, 4.74033833, 6.32958761, 7.14991049,
                     14.54004725, 9.00723817, 19.77900076, 14.48882752, 11.67445804,
                     3.43512478, 14.49896383, 3.86199652, 4.40980132, 7.88499867,
                     1.05552782, 4.26679215, 12.84775443, 19.14961564, 11.86131258,
                     11.68877294, 1.41722509, 9.85780501, 12.78118287, 9.48000153,
                     17.454911, 5.27828154, 6.39579851, 5.38761666, 5.05070064,
                     65.23083872, 7.95510104, 7.11598887, 20.66733863, 21.67808584,
                     12.63920315, 53.48396713, 84.48455992, 10.73392297, 59.32004465,
                     99.60248082, 94.73063131, 81.97253032, 54.97999684, 16.76602762,
                     74.68253242, 47.86843811, 96.52613688, 15.16096189, 18.7041231])

    actual = PyChopModel._get_he_detector_width_squared(data)
    expected = np.array([tube_mts(val)[3] for val in data])

    assert_allclose(actual, expected)


@pytest.mark.parametrize('e_init', EINIT, ids=format_ei)
def test_fermi_detector_width_squared(e_init, pychop_fermi_data):
    _test_get_detector_width_squared(e_init, PyChopModelFermi, *pychop_fermi_data)


@pytest.mark.parametrize('e_init', EINIT, ids=format_ei)
def test_nonfermi_detector_width_squared(e_init, pychop_nonfermi_data):
    _test_get_detector_width_squared(e_init, PyChopModelNonFermi, *pychop_nonfermi_data)


def _test_get_detector_width_squared(e_init, cls, data, pychop):
    if data.detector is None:
        pytest.skip('This instrument does not have a detector defined')

    fake_frequencies = get_fake_frequencies(e_init)

    actual = cls._get_detector_width_squared(data.detector, fake_frequencies, e_init)
    expected = np.array([pychop.detector.getWidthSquared(e_init, en) for en in fake_frequencies])

    assert_allclose(actual, expected)


def test_fermi_sample_width_squared(pychop_fermi_data):
    pychop_fermi_data, pychop = pychop_fermi_data

    actual = PyChopModelFermi._get_sample_width_squared(pychop_fermi_data.sample)
    expected = pychop.sample.getWidthSquared()

    assert_allclose(actual, expected)


def test_nonfermi_sample_width_squared(pychop_nonfermi_data):
    data, pychop = pychop_nonfermi_data

    if data.detector is None:
        pytest.skip('This instrument does not have a detector defined')

    actual = PyChopModelFermi._get_sample_width_squared(data.sample)
    expected = pychop.sample.getWidthSquared()

    assert_allclose(actual, expected)


@pytest.mark.skipif(DEBUG, reason='Debugging precompute_van_var; its outputs have been temporarily changed.')
@pytest.mark.parametrize('matrix', MATRIX_FERMI, ids=matrix_fermi_id)
def test_fermi_precompute_van_var(matrix, pychop_fermi_data):
    e_init, chopper_frequency = matrix
    _test_precompute_van_var(e_init, [chopper_frequency], PyChopModelFermi, *pychop_fermi_data)


@pytest.mark.skipif(DEBUG, reason='Debugging precompute_van_var; its outputs have been temporarily changed.')
@pytest.mark.parametrize('matrix', MATRIX_NONFERMI, ids=matrix_nonfermi_id)
def test_nonfermi_precompute_van_var(matrix, pychop_nonfermi_data):
    e_init, *chopper_frequency = matrix
    _test_precompute_van_var(e_init, chopper_frequency, PyChopModelNonFermi, *pychop_nonfermi_data)


def _test_precompute_van_var(e_init, chopper_frequency, cls, data, pychop):
    fake_frequencies = get_fake_frequencies(e_init)

    try:
        pychop.chopper_system.setFrequency(chopper_frequency)
    except ValueError as e:
        if 'maximum allowed' in str(e):
            pytest.skip('Frequency outside the bounds of this instrument')
    expected, _, _ = pychop.getVanVar(Ei_in=e_init, Etrans=fake_frequencies)

    if np.any(np.isnan(expected)):
        with pytest.raises(NoTransmissionError):
            cls._precompute_van_var(data, e_init, list(chopper_frequency), fake_frequencies)

    else:
        actual = cls._precompute_van_var(data, e_init, list(chopper_frequency), fake_frequencies)

        assert_allclose(actual, expected)


@pytest.mark.skipif(not DEBUG, reason='Not debugging; normal version of the function only returns vsq_van')
@pytest.mark.parametrize('matrix', MATRIX_FERMI, ids=matrix_fermi_id)
def test_debug_fermi_precompute_van_var(matrix, pychop_fermi_data):
    e_init, chopper_frequency = matrix
    _test_debug_precompute_van_var(e_init, [chopper_frequency], PyChopModelFermi, *pychop_fermi_data)


@pytest.mark.skipif(not DEBUG, reason='Not debugging; normal version of the function only returns vsq_van')
@pytest.mark.parametrize('matrix', MATRIX_NONFERMI, ids=matrix_nonfermi_id)
def test_debug_nonfermi_precompute_van_var(matrix, pychop_nonfermi_data):
    e_init, *chopper_frequency = matrix
    _test_debug_precompute_van_var(e_init, chopper_frequency, PyChopModelNonFermi, *pychop_nonfermi_data)


def _test_debug_precompute_van_var(e_init, chopper_frequency, cls, data, pychop):
    fake_frequencies = get_fake_frequencies(e_init)

    try:
        pychop.chopper_system.setFrequency(chopper_frequency)
    except ValueError as e:
        if 'maximum allowed' in str(e):
            pytest.skip('Frequency outside the bounds of this instrument')
    expected_result, expected, _ = pychop.getVanVar(Ei_in=e_init, Etrans=fake_frequencies)

    if np.any(np.isnan(expected_result)):
        with pytest.raises(NoTransmissionError):
            cls._precompute_resolution(data, e_init, list(chopper_frequency))

    else:
        vsq_van, tsq_moderator, tsq_chopper, tsq_jit, tsq_aperture, *other_tsq = \
            cls._precompute_van_var(data, e_init, list(chopper_frequency), fake_frequencies)

        assert_allclose(tsq_moderator, expected['moderator'])
        assert_allclose(tsq_chopper, expected['chopper'])
        assert_allclose(tsq_jit, expected['jitter'])
        assert_allclose(tsq_aperture, expected['aperture'])

        if data.detector is not None:
            tsq_detector = other_tsq.pop(0)
            assert_allclose(tsq_detector, expected['detector'])

        if data.sample is not None:
            assert_allclose(other_tsq[0], expected['sample'])

        assert_allclose(vsq_van, expected_result)


@pytest.mark.skipif(DEBUG, reason='Debugging precompute_van_var; its outputs have been temporarily changed.')
@pytest.mark.parametrize('matrix', MATRIX_FERMI, ids=matrix_fermi_id)
def test_fermi_precompute_resolution(matrix, pychop_fermi_data):
    e_init, chopper_frequency = matrix
    _test_precompute_resolution(e_init, [chopper_frequency], PyChopModelFermi, *pychop_fermi_data)


@pytest.mark.skipif(DEBUG, reason='Debugging precompute_van_var; its outputs have been temporarily changed.')
@pytest.mark.parametrize('matrix', MATRIX_NONFERMI, ids=matrix_nonfermi_id)
def test_nonfermi_precompute_resolution(matrix, pychop_nonfermi_data):
    e_init, *chopper_frequency = matrix
    _test_precompute_resolution(e_init, chopper_frequency, PyChopModelNonFermi, *pychop_nonfermi_data)


def _test_precompute_resolution(e_init, chopper_frequency, cls, data, pychop):
    try:
        pychop.chopper_system.setFrequency(chopper_frequency)
    except ValueError as e:
        if 'Value of frequencies outside maximum allowed' in str(e):
            # Energy out of range for Pychop: make sure the model agrees
            with pytest.raises(
                    InvalidInputError,
                    match=rf"The provided chopper frequency \(\[{chopper_frequency[0]}\]\) is not allowed"):
                cls(model_data=data, chopper_frequency=chopper_frequency, e_init=e_init)

            return
        raise e

    fake_frequencies = np.linspace(0, e_init, 40, endpoint=False)
    expected_resolution = pychop.getResolution(Ei_in=e_init, Etrans=fake_frequencies)

    if np.any(np.isnan(expected_resolution)):
        with pytest.raises(NoTransmissionError):
            cls._precompute_resolution(data, e_init, list(chopper_frequency))

    else:
        actual = cls._precompute_resolution(data, e_init, list(chopper_frequency))

        assert_allclose(actual[1], expected_resolution / SIGMA2FWHM)
