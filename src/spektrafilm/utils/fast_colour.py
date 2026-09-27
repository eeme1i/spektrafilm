"""Faster drop-ins for colour's RGB <-> XYZ conversions on large images.

``colour.RGB_to_XYZ`` / ``XYZ_to_RGB`` / ``RGB_to_RGB`` are linear maps
(plus an optional CCTF), but colour applies each 3x3 matrix through
``vecmul``, i.e. millions of tiny batched matmuls. Here the composed
3x3 matrix is obtained from colour itself (by transforming the identity,
so primaries, whitepoints and chromatic adaptation are exactly colour's)
and applied to the image with a single BLAS matmul. Results agree with
colour to floating-point rounding.

The common transfer functions (sRGB, gamma, BT.2020, ROMM) are likewise
recognised and evaluated by one parallel numba pass using colour's exact
piecewise definitions and constants; any other CCTF goes through colour.
"""

import functools

import colour
import numpy as np
from colour.models.rgb.transfer_functions import (
    cctf_decoding_ROMMRGB,
    cctf_encoding_ROMMRGB,
    eotf_inverse_sRGB,
    eotf_sRGB,
    gamma_function,
    linear_function,
    oetf_BT2020,
    oetf_inverse_BT2020,
)
from colour.models.rgb.transfer_functions.itur_bt_2020 import CONSTANTS_BT2020
from numba import njit, prange

from spektrafilm.utils.dtypes import as_float_array

_MATRIX_CACHE: dict[tuple, np.ndarray] = {}


def _colourspace(colourspace):
    if isinstance(colourspace, str):
        return colour.RGB_COLOURSPACES[colourspace]
    return colourspace


def _key(*parts):
    key = []
    for part in parts:
        if part is None or isinstance(part, str):
            key.append(part)
        elif isinstance(part, colour.RGB_Colourspace):
            return None  # not reliably hashable; recompute (a 3x3 is cheap)
        else:
            key.append(tuple(np.asarray(part, dtype=float).ravel()))
    return tuple(key)


def _cached_matrix(key, compute):
    if key is None:
        return compute()
    matrix = _MATRIX_CACHE.get(key)
    if matrix is None:
        matrix = compute()
        _MATRIX_CACHE[key] = matrix
    return matrix


_SRGB_ENCODE, _SRGB_DECODE, _GAMMA, _BT2020_ENCODE, _BT2020_DECODE, _ROMM_ENCODE, _ROMM_DECODE = range(7)
_ROMM_I_MAX = 2**8 - 1
_ROMM_E_T = 16 ** (1.8 / (1 - 1.8))


@njit(inline='always', cache=True)
def _spow(x, p):
    # colour.algebra.spow: sign-preserving power.
    if x < 0.0:
        return -((-x) ** p)
    return x ** p


@njit(parallel=True, cache=True)
def _cctf_kernel(flat, out, mode, p0, p1, p2):
    # Each branch mirrors the np.where of the colour function it replaces
    # (same comparison, so NaN takes the same branch).
    for i in prange(flat.shape[0]):
        x = float(flat[i])
        if mode == _SRGB_ENCODE:
            y = x * 12.92 if x <= 0.0031308 else 1.055 * _spow(x, 1 / 2.4) - 0.055
        elif mode == _SRGB_DECODE:
            y = x / 12.92 if p0 >= x else _spow((x + 0.055) / 1.055, 2.4)
        elif mode == _GAMMA:
            # 'Indeterminate' negative handling: plain power, as NumPy's.
            y = x ** p0
        elif mode == _BT2020_ENCODE:
            y = x * 4.5 if p1 > x else p0 * _spow(x, 0.45) - (p0 - 1)
        elif mode == _BT2020_DECODE:
            y = x / 4.5 if x < p2 else _spow((x + (p0 - 1)) / p0, 1 / 0.45)
        elif mode == _ROMM_ENCODE:
            if _ROMM_E_T > x:
                y = x * 16 * _ROMM_I_MAX / _ROMM_I_MAX
            else:
                y = _spow(x, 1 / 1.8) * _ROMM_I_MAX / _ROMM_I_MAX
        else:  # _ROMM_DECODE
            x_p = x * _ROMM_I_MAX
            if x_p < 16 * _ROMM_E_T * _ROMM_I_MAX:
                y = x_p / (16 * _ROMM_I_MAX)
            else:
                y = _spow(x_p / _ROMM_I_MAX, 1.8)
        out[i] = y


def _fast_cctf_spec(function):
    """(mode, p0, p1, p2) for a CCTF the numba kernel reproduces, 'linear'
    for the identity, or None to defer to colour."""
    if colour.get_domain_range_scale() != 'reference':
        return None
    keywords = {}
    if isinstance(function, functools.partial):
        if function.args:
            return None
        keywords = dict(function.keywords)
        function = function.func
    if function is linear_function and not keywords:
        return 'linear'
    if function is gamma_function:
        if set(keywords) != {'exponent'} or np.ndim(keywords['exponent']) != 0:
            return None
        return (_GAMMA, float(keywords['exponent']), 0.0, 0.0)
    if keywords:
        return None
    if function is eotf_inverse_sRGB:
        return (_SRGB_ENCODE, 0.0, 0.0, 0.0)
    if function is eotf_sRGB:
        return (_SRGB_DECODE, float(eotf_inverse_sRGB(0.0031308)), 0.0, 0.0)
    a = float(CONSTANTS_BT2020.alpha(False))
    b = float(CONSTANTS_BT2020.beta(False))
    if function is oetf_BT2020:
        return (_BT2020_ENCODE, a, b, 0.0)
    if function is oetf_inverse_BT2020:
        return (_BT2020_DECODE, a, b, float(oetf_BT2020(b)))
    if function is cctf_encoding_ROMMRGB:
        return (_ROMM_ENCODE, 0.0, 0.0, 0.0)
    if function is cctf_decoding_ROMMRGB:
        return (_ROMM_DECODE, 0.0, 0.0, 0.0)
    return None


def _cctf(function, values):
    # Results come back in the input's working precision so float32 images
    # stay float32 (colour itself evaluates in float64).
    values = as_float_array(values)
    spec = _fast_cctf_spec(function)
    if spec is None:
        return np.asarray(function(values)).astype(values.dtype, copy=False)
    if spec == 'linear':
        return values
    flat = np.ascontiguousarray(values).reshape(-1)
    out = np.empty_like(flat)
    _cctf_kernel(flat, out, *spec)
    return out.reshape(values.shape)


def warmup_fast_colour():
    for dtype in (np.float64, np.float32):
        values = np.linspace(-0.1, 1.1, 8, dtype=dtype)
        _cctf(eotf_inverse_sRGB, values)


def _apply_matrix(values, matrix_transposed):
    values = as_float_array(values)
    flat = values.reshape(-1, 3)
    return (flat @ matrix_transposed.astype(values.dtype, copy=False)).reshape(values.shape)


def rgb_to_xyz(rgb, colourspace, illuminant=None, chromatic_adaptation_transform='CAT02',
               apply_cctf_decoding=False):
    """Fast equivalent of ``colour.RGB_to_XYZ`` (same arguments)."""
    matrix_t = _cached_matrix(
        _key('RGB_to_XYZ', colourspace, illuminant, chromatic_adaptation_transform),
        lambda: np.asarray(colour.RGB_to_XYZ(
            np.eye(3), colourspace, illuminant=illuminant,
            chromatic_adaptation_transform=chromatic_adaptation_transform,
        )),
    )
    if apply_cctf_decoding:
        rgb = _cctf(_colourspace(colourspace).cctf_decoding, rgb)
    return _apply_matrix(rgb, matrix_t)


def xyz_to_rgb(xyz, colourspace, illuminant=None, chromatic_adaptation_transform='CAT02',
               apply_cctf_encoding=False):
    """Fast equivalent of ``colour.XYZ_to_RGB`` (same arguments)."""
    matrix_t = _cached_matrix(
        _key('XYZ_to_RGB', colourspace, illuminant, chromatic_adaptation_transform),
        lambda: np.asarray(colour.XYZ_to_RGB(
            np.eye(3), colourspace, illuminant=illuminant,
            chromatic_adaptation_transform=chromatic_adaptation_transform,
        )),
    )
    rgb = _apply_matrix(xyz, matrix_t)
    if apply_cctf_encoding:
        rgb = _cctf(_colourspace(colourspace).cctf_encoding, rgb)
    return rgb


def rgb_to_rgb(rgb, input_colourspace, output_colourspace, chromatic_adaptation_transform='CAT02',
               apply_cctf_decoding=False, apply_cctf_encoding=False):
    """Fast equivalent of ``colour.RGB_to_RGB`` (same arguments)."""
    matrix_t = _cached_matrix(
        _key('RGB_to_RGB', input_colourspace, output_colourspace, chromatic_adaptation_transform),
        lambda: np.asarray(colour.RGB_to_RGB(
            np.eye(3), input_colourspace, output_colourspace,
            chromatic_adaptation_transform=chromatic_adaptation_transform,
        )),
    )
    if apply_cctf_decoding:
        rgb = _cctf(_colourspace(input_colourspace).cctf_decoding, rgb)
    rgb = _apply_matrix(rgb, matrix_t)
    if apply_cctf_encoding:
        rgb = _cctf(_colourspace(output_colourspace).cctf_encoding, rgb)
    return rgb
