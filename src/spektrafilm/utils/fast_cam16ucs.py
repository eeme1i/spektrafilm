"""Numba CAM16-UCS forward / inverse for fixed viewing conditions.

Drop-in replacements for ``colour.XYZ_to_CAM16UCS`` and
``colour.CAM16UCS_to_XYZ`` (colour-science's default "reference"
domain-range scale: XYZ at Y=1, Jpapbp at 100). colour's generic
implementation evaluates every CAM16 correlate (Q, s, H, ...) through
array helpers and is the single most expensive step of output gamut
compression; here only J, M, h are computed, per pixel, in parallel.

The per-pixel math mirrors colour's step by step, including its
``sdiv`` "ignore zero conversion" (non-finite quotients become 0) and
``spow`` (sign-preserving power) conventions, so results agree with
colour to floating-point rounding. The whitepoint-dependent constants
are derived with colour's own helpers.
"""

import math

import numpy as np
from numba import njit, prange

_EPSILON = float(np.finfo(np.float64).eps)
_UCS_C1 = 0.007
_UCS_C2 = 0.0228

_CONSTANTS_CACHE: dict[tuple, tuple] = {}


def _cam16_constants(xyz_w, L_A, Y_b):
    key = (tuple(np.asarray(xyz_w, dtype=float).ravel()), float(L_A), float(Y_b))
    cached = _CONSTANTS_CACHE.get(key)
    if cached is not None:
        return cached

    from colour.appearance import CAM_KWARGS_CIECAM02_sRGB
    from colour.appearance.cam16 import MATRIX_16, MATRIX_INVERSE_16
    from colour.appearance.ciecam02 import (
        achromatic_response_forward,
        degree_of_adaptation,
        post_adaptation_non_linear_response_compression_forward,
        viewing_conditions_dependent_parameters,
    )

    surround = CAM_KWARGS_CIECAM02_sRGB["surround"]
    XYZ_w = np.asarray(xyz_w, dtype=float) * 100
    Y_w = XYZ_w[1]
    L_A = np.asarray(L_A, dtype=float)
    RGB_w = MATRIX_16 @ XYZ_w
    D = np.clip(degree_of_adaptation(surround.F, L_A), 0, 1)
    n, F_L, N_bb, N_cb, z = viewing_conditions_dependent_parameters(Y_b, Y_w, L_A)
    D_RGB = D * Y_w / RGB_w + 1 - D
    RGB_aw = post_adaptation_non_linear_response_compression_forward(D_RGB * RGB_w, F_L)
    A_w = achromatic_response_forward(RGB_aw, N_bb)

    constants = (
        np.ascontiguousarray(MATRIX_16, dtype=np.float64),
        np.ascontiguousarray(MATRIX_INVERSE_16, dtype=np.float64),
        np.ascontiguousarray(D_RGB, dtype=np.float64),
        float(F_L),
        float(N_bb),
        float(A_w),
        float(surround.c * z),
        float(50000 / 13 * surround.N_c * N_cb),
        float((1.64 - 0.29**n) ** 0.73),
        float(F_L**0.25),
    )
    _CONSTANTS_CACHE[key] = constants
    return constants


@njit(cache=True, inline='always')
def _spow(a, p):
    if a > 0.0:
        return a**p
    if a < 0.0:
        return -((-a) ** p)
    return a  # 0 or nan


@njit(cache=True, inline='always', error_model='numpy')
def _zdiv(a, b):
    c = a / b
    if math.isfinite(c):
        return c
    return 0.0


@njit(cache=True, inline='always')
def _sign(a):
    if a > 0.0:
        return 1.0
    if a < 0.0:
        return -1.0
    return a  # 0 or nan


@njit(parallel=True, cache=True, error_model='numpy')
def _xyz_to_cam16ucs_kernel(xyz, M16, D_RGB, F_L, N_bb, A_w, cz, t_coef, chroma_k, F_L_025, out):
    for i in prange(xyz.shape[0]):
        X = xyz[i, 0] * 100.0
        Y = xyz[i, 1] * 100.0
        Z = xyz[i, 2] * 100.0
        Ra = 0.0
        Ga = 0.0
        Ba = 0.0
        for ch in range(3):
            rc = (M16[ch, 0] * X + M16[ch, 1] * Y + M16[ch, 2] * Z) * D_RGB[ch]
            f = (F_L * abs(rc) / 100.0) ** 0.42
            ra = 400.0 * _sign(rc) * f / (27.13 + f) + 0.1
            if ch == 0:
                Ra = ra
            elif ch == 1:
                Ga = ra
            else:
                Ba = ra

        a = Ra - 12.0 * Ga / 11.0 + Ba / 11.0
        b = (Ra + Ga - 2.0 * Ba) / 9.0
        h = math.degrees(math.atan2(b, a)) % 360.0
        e_t = 1.0 / 4.0 * (math.cos(2.0 + h * math.pi / 180.0) + 3.8)
        if h != h:
            h = 0.0  # colour's hue_quadrature zeroes nan hues in place
        A = (2.0 * Ra + Ga + 1.0 / 20.0 * Ba - 0.305) * N_bb
        J = 100.0 * _spow(_zdiv(A, A_w), cz)
        t = t_coef * _zdiv(e_t * _spow(a * a + b * b, 0.5), Ra + Ga + 21.0 * Ba / 20.0)
        C = _spow(t, 0.9) * _spow(J / 100.0, 0.5) * chroma_k
        M = C * F_L_025

        J_p = (1.0 + 100.0 * _UCS_C1) * J / (1.0 + _UCS_C1 * J)
        M_p = 1.0 / _UCS_C2 * math.log1p(_UCS_C2 * M)
        hr = math.radians(h)
        out[i, 0] = J_p
        out[i, 1] = M_p * math.cos(hr)
        out[i, 2] = M_p * math.sin(hr)
    return out


@njit(parallel=True, cache=True, error_model='numpy')
def _cam16ucs_to_xyz_kernel(jab, M16_inv, D_RGB, F_L, N_bb, A_w, cz, t_coef, chroma_k, F_L_025, out):
    for i in prange(jab.shape[0]):
        J_p = jab[i, 0]
        a_p = jab[i, 1]
        b_p = jab[i, 2]
        J = -J_p / (_UCS_C1 * J_p - 1.0 - 100.0 * _UCS_C1)
        M_p = math.hypot(a_p, b_p)
        h = math.degrees(math.atan2(b_p, a_p)) % 360.0
        M = math.expm1(M_p / (1.0 / _UCS_C2)) / _UCS_C2

        C = M / F_L_025
        J_floor = J if (J > _EPSILON or J != J) else _EPSILON
        t = _spow(C / (math.sqrt(J_floor / 100.0) * chroma_k), 1.0 / 0.9)
        e_t = 1.0 / 4.0 * (math.cos(2.0 + h * math.pi / 180.0) + 3.8)
        A = A_w * _spow(J / 100.0, 1.0 / cz)
        P_1 = _zdiv(t_coef * e_t, t)
        P_2 = A / N_bb + 0.305
        P_3 = 21.0 / 20.0

        hr = math.radians(h)
        sin_hr = math.sin(hr)
        cos_hr = math.cos(hr)
        n = P_2 * (2.0 + P_3) * (460.0 / 1403.0)
        a = 0.0
        b = 0.0
        if abs(sin_hr) >= abs(cos_hr):
            cos_hr_sin_hr = _zdiv(cos_hr, sin_hr)
            P_4 = _zdiv(P_1, sin_hr)
            b = n / (P_4 + (2.0 + P_3) * (220.0 / 1403.0) * cos_hr_sin_hr
                     - 27.0 / 1403.0 + P_3 * (6300.0 / 1403.0))
            a = b * cos_hr_sin_hr
        elif abs(sin_hr) < abs(cos_hr):
            sin_hr_cos_hr = _zdiv(sin_hr, cos_hr)
            P_5 = _zdiv(P_1, cos_hr)
            a = n / (P_5 + (2.0 + P_3) * (220.0 / 1403.0)
                     - (27.0 / 1403.0 - P_3 * (6300.0 / 1403.0)) * sin_hr_cos_hr)
            b = a * sin_hr_cos_hr
        if t == 0.0:
            a = a * 0.0
            b = b * 0.0

        R0 = (460.0 * P_2 + 451.0 * a + 288.0 * b) / 1403.0
        R1 = (460.0 * P_2 - 891.0 * a - 261.0 * b) / 1403.0
        R2 = (460.0 * P_2 - 220.0 * a - 6300.0 * b) / 1403.0

        rgb0 = 0.0
        rgb1 = 0.0
        rgb2 = 0.0
        for ch in range(3):
            if ch == 0:
                x = R0 - 0.1
            elif ch == 1:
                x = R1 - 0.1
            else:
                x = R2 - 0.1
            ax = abs(x)
            rc = _sign(x) * 100.0 / F_L * _spow(27.13 * ax / (400.0 - ax), 1.0 / 0.42)
            rc = rc / D_RGB[ch]
            if ch == 0:
                rgb0 = rc
            elif ch == 1:
                rgb1 = rc
            else:
                rgb2 = rc

        for ch in range(3):
            out[i, ch] = (M16_inv[ch, 0] * rgb0 + M16_inv[ch, 1] * rgb1 + M16_inv[ch, 2] * rgb2) / 100.0
    return out


def xyz_to_cam16ucs(xyz, xyz_w, L_A, Y_b):
    """Fast equivalent of ``colour.XYZ_to_CAM16UCS(xyz, XYZ_w=xyz_w, L_A=L_A, Y_b=Y_b)``."""
    M16, _, D_RGB, *scalars = _cam16_constants(xyz_w, L_A, Y_b)
    xyz = np.asarray(xyz, dtype=np.float64)
    flat = np.ascontiguousarray(xyz.reshape(-1, 3))
    out = np.empty_like(flat)
    _xyz_to_cam16ucs_kernel(flat, M16, D_RGB, *scalars, out)
    return out.reshape(xyz.shape)


def cam16ucs_to_xyz(jab, xyz_w, L_A, Y_b):
    """Fast equivalent of ``colour.CAM16UCS_to_XYZ(jab, XYZ_w=xyz_w, L_A=L_A, Y_b=Y_b)``."""
    _, M16_inv, D_RGB, *scalars = _cam16_constants(xyz_w, L_A, Y_b)
    jab = np.asarray(jab, dtype=np.float64)
    flat = np.ascontiguousarray(jab.reshape(-1, 3))
    out = np.empty_like(flat)
    _cam16ucs_to_xyz_kernel(flat, M16_inv, D_RGB, *scalars, out)
    return out.reshape(jab.shape)


def warmup_fast_cam16ucs():
    xyz_w = np.array([0.95047, 1.0, 1.08883])
    xyz = np.array([[[0.2, 0.3, 0.4]]])
    cam16ucs_to_xyz(xyz_to_cam16ucs(xyz, xyz_w, 64.0, 20.0), xyz_w, 64.0, 20.0)
