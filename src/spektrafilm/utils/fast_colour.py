"""Faster drop-ins for colour's RGB <-> XYZ conversions on large images.

``colour.RGB_to_XYZ`` / ``XYZ_to_RGB`` / ``RGB_to_RGB`` are linear maps
(plus an optional CCTF), but colour applies each 3x3 matrix through
``vecmul``, i.e. millions of tiny batched matmuls. Here the composed
3x3 matrix is obtained from colour itself (by transforming the identity,
so primaries, whitepoints and chromatic adaptation are exactly colour's)
and applied to the image with a single BLAS matmul. Results agree with
colour to floating-point rounding.
"""

import colour
import numpy as np

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


def _cctf(function, values):
    # colour evaluates transfer functions in float64; return the input's
    # working precision so float32 images stay float32.
    values = as_float_array(values)
    return np.asarray(function(values)).astype(values.dtype, copy=False)


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
