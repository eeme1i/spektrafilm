import numpy as np
import colour
from numba import njit, prange
from opt_einsum import contract

from spektrafilm.config import SPECTRAL_SHAPE


@njit(parallel=True, cache=True)
def _cmy_density_to_weighted_light_kernel(density_cmy, channel_density, base_density, light, weights, out):
    n_wavelengths = channel_density.shape[0]
    for i in prange(density_cmy.shape[0]):
        c = density_cmy[i, 0]
        m = density_cmy[i, 1]
        y = density_cmy[i, 2]
        acc0 = 0.0
        acc1 = 0.0
        acc2 = 0.0
        for k in range(n_wavelengths):
            density = c * channel_density[k, 0] + m * channel_density[k, 1] + y * channel_density[k, 2] + base_density[k]
            transmitted = 10.0 ** (-density) * light[k]
            if np.isnan(transmitted):
                transmitted = 0.0
            acc0 += transmitted * weights[k, 0]
            acc1 += transmitted * weights[k, 1]
            acc2 += transmitted * weights[k, 2]
        out[i, 0] = acc0
        out[i, 1] = acc1
        out[i, 2] = acc2
    return out


def cmy_density_to_weighted_light(density_cmy, channel_density, base_density, light, weights):
    """Fused equivalent of
    ``contract('ijk,kl->ijl', density_to_light(compute_density_spectral(channel_density, density_cmy, base_density), light), weights)``.

    Integrates the light transmitted through the dye layers against three
    spectral weighting functions (e.g. print sensitivities or CMFs) per pixel,
    without materializing the (..., n_wavelengths) spectral density array.
    """
    density_cmy = np.asarray(density_cmy, dtype=np.float64)
    channel_density = np.ascontiguousarray(channel_density, dtype=np.float64)
    n_wavelengths = channel_density.shape[0]
    if base_density is None:
        base_density = np.zeros(n_wavelengths)
    base_density = np.ascontiguousarray(np.broadcast_to(base_density, (n_wavelengths,)), dtype=np.float64)
    light = np.ascontiguousarray(np.broadcast_to(light, (n_wavelengths,)), dtype=np.float64)
    weights = np.ascontiguousarray(weights, dtype=np.float64)
    flat = np.ascontiguousarray(density_cmy.reshape(-1, 3))
    out = np.empty_like(flat)
    _cmy_density_to_weighted_light_kernel(flat, channel_density, base_density, light, weights, out)
    return out.reshape(density_cmy.shape)


def warmup_conversions():
    cmy_density_to_weighted_light(np.zeros((1, 1, 3)), np.ones((4, 3)), np.zeros(4), np.ones(4), np.ones((4, 3)))


def density_to_light(density, light):
    """
    Convert density to light transmittance.

    This function calculates the light transmittance based on the given density
    and light intensity. It uses the formula transmittance = 10^(-density) to 
    compute the transmittance and then multiplies it by the light intensity.

    Parameters:
    density (float or np.ndarray): The density value(s) which affect the light transmittance.
    light (float or np.ndarray): The initial light intensity value(s).

    Returns:
    np.ndarray: The light intensity after passing through the medium with the given density.
    """
    transmitted = 10**(-density)
    transmitted *= light
    transmitted[np.isnan(transmitted)] = 0
    return transmitted

def compute_aces_conversion_matrix(sensitivity, illuminant):            
    """
    Computes the ACES (Academy Color Encoding System) conversion matrix.

    Parameters
    ----------
    sensitivity : array-like
        The spectral sensitivity data.
    illuminant : array-like
        The illuminant spectral distribution.

    Returns
    -------
    numpy.ndarray
        The ACES to raw conversion matrix.
    """
    msds = colour.MultiSpectralDistributions(sensitivity, domain=SPECTRAL_SHAPE.wavelengths)
    M, _ = colour.matrix_idt(msds, illuminant)
    aces_to_raw_conversion_matrix = np.linalg.inv(M)
    return aces_to_raw_conversion_matrix

def rgb_to_raw_aces_idt(RGB, illuminant, sensitivity, midgray_rgb=[[[0.184,0.184,0.184]]],
                        color_space='sRGB', apply_cctf_decoding=True, aces_conversion_matrix=[]):
    """
    Converts RGB values to raw values using ACES IDT (Input Device Transform).

    Parameters:
    RGB (array-like): The input RGB values.
    illuminant (array-like): The illuminant data.
    sensitivity (array-like): The sensitivity data.
    midgray_rgb (array-like, optional): The mid-gray RGB values. Default is [[[0.184, 0.184, 0.184]]].
    color_space (str, optional): The color space of the input RGB values. Default is 'sRGB'.
    apply_cctf_decoding (bool, optional): Whether to apply the CCTF decoding. Default is True.
    aces_conversion_matrix (array-like, optional): The ACES conversion matrix. Default is an empty list.

    Returns:
    tuple: A tuple containing:
        - raw (array-like): The raw values.
        - raw_midgray (array-like): The raw mid-gray values.
    """
    aces = colour.RGB_to_RGB(RGB, color_space, 'ACES2065-1',
                    apply_cctf_decoding=apply_cctf_decoding,
                    apply_cctf_encoding=False)
    if aces_conversion_matrix==[]:
        aces_conversion_matrix = compute_aces_conversion_matrix(sensitivity, illuminant)
    raw = contract('ijk,lk->ijl',aces,aces_conversion_matrix)/midgray_rgb
    raw_midgray = np.array([[[1,1,1]]])
    return raw, raw_midgray

