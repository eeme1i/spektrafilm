from __future__ import annotations

import functools

import colour
import numpy as np
import pytest

from colour.models.rgb.transfer_functions import gamma_function
from spektrafilm.utils import fast_colour


pytestmark = pytest.mark.unit

GUI_COLOR_SPACES = ('sRGB', 'DCI-P3', 'Display P3', 'Adobe RGB (1998)',
                    'ITU-R BT.2020', 'ProPhoto RGB', 'ACES2065-1')


def _probe_values() -> np.ndarray:
    # Dense range plus the piecewise breakpoints and non-finite values.
    return np.concatenate([
        np.linspace(-0.5, 1.5, 20001),
        [0.0, 0.0031308, 0.04045, 0.018, 0.081, 0.001953125, 0.03125, 1e-12,
         np.nan, np.inf, -np.inf],
    ])


@pytest.mark.parametrize('direction', ['cctf_encoding', 'cctf_decoding'])
@pytest.mark.parametrize('name', GUI_COLOR_SPACES)
def test_fast_cctf_matches_colour(name, direction) -> None:
    function = getattr(colour.RGB_COLOURSPACES[name], direction)
    values = _probe_values()

    assert fast_colour._fast_cctf_spec(function) is not None
    with np.errstate(invalid='ignore'):
        expected = np.asarray(function(values.copy()))
    np.testing.assert_allclose(fast_colour._cctf(function, values), expected, rtol=1e-15, atol=0)


def test_fast_cctf_keeps_float32() -> None:
    function = colour.RGB_COLOURSPACES['sRGB'].cctf_encoding
    values = np.linspace(0.0, 1.0, 101, dtype=np.float32)

    result = fast_colour._cctf(function, values)

    assert result.dtype == np.float32
    np.testing.assert_allclose(result, function(values.astype(np.float64)), rtol=1e-6)


def test_unrecognised_cctf_falls_back_to_colour() -> None:
    function = functools.partial(gamma_function, exponent=2.2, negative_number_handling='Mirror')
    values = np.array([-0.5, 0.25, 1.0])

    assert fast_colour._fast_cctf_spec(function) is None
    np.testing.assert_allclose(fast_colour._cctf(function, values), function(values))
