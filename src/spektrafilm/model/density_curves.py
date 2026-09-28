import numba
import numpy as np
import scipy
from spektrafilm.utils.dtypes import as_float_array
from spektrafilm.utils.fast_interp import fast_interp

################################################################################
# Denstity curves
################################################################################

def interpolate_exposure_to_density(log_exposure_rgb, density_curves, log_exposure, gamma_factor):
    """
    Interpolates the exposure values to density values using the provided density curves.
    Parameters:
    log_exposure_rgb (numpy.ndarray): A 3D array of shape (height, width, 3) representing the log10 RGB exposure values.
    density_curves (numpy.ndarray): A 2D array of shape (num_points, 3) representing the density curves for each channel.
    log_exposure (numpy.ndarray): A 1D array of logarithmic exposure values.
    gamma_factor (float): The gamma correction factor to be applied to the density characteristic curves.
    Returns:
    numpy.ndarray: A 3D array of shape (height, width, 3) representing the interpolated density values in CMY channels.
    """
    if np.size(gamma_factor)==1:
        gamma_factor = [gamma_factor, gamma_factor, gamma_factor]
    gamma_factor = np.array(gamma_factor)
    density_cmy = np.zeros((log_exposure_rgb.shape[0], log_exposure_rgb.shape[1], 3))
    # for channel in np.arange(3):
    #     sel = ~np.isnan(density_curves[:,channel])
    #     density_cmy[:,:,channel] = np.interp(log_exposure_rgb[:,:,channel],
    #                                          log_exposure[sel]/gamma_factor[channel],
    #                                          density_curves[sel,channel])
    density_cmy = fast_interp(np.ascontiguousarray(log_exposure_rgb),
                              log_exposure[:,None]/gamma_factor[None,:],
                              density_curves)
    return density_cmy


@numba.njit(parallel=True, fastmath=True, cache=True)
def _interp_density_layer_kernel(density, x_axis, inv_dx, y_layer, sign, out):
    # Same interpolation rule as fast_interp: endpoint clamping, right-biased exact matches.
    n_rows, n_cols = density.shape
    K = x_axis.shape[0]
    for i in numba.prange(n_rows):
        for j in range(n_cols):
            x = sign * density[i, j]
            if x <= x_axis[0]:
                out[i, j] = y_layer[0]
            elif x >= x_axis[K - 1]:
                out[i, j] = y_layer[K - 1]
            else:
                low = np.searchsorted(x_axis, x, side='right') - 1
                t = (x - x_axis[low]) * inv_dx[low]
                out[i, j] = y_layer[low] + t * (y_layer[low + 1] - y_layer[low])
    return out


@numba.njit(parallel=True, fastmath=True, cache=True)
def _interp_density_channel_layers_kernel(density, x_axis, inv_dx, y_layers, sign, channel):
    """Interpolate all three sublayers with one density-axis lookup per pixel."""
    n_rows, n_cols = density.shape[:2]
    out = np.empty((3, n_rows, n_cols), dtype=density.dtype)
    K = x_axis.shape[1]
    for i in numba.prange(n_rows):
        for j in range(n_cols):
            x = sign * density[i, j, channel]
            if x <= x_axis[channel, 0]:
                for layer in range(3):
                    out[layer, i, j] = y_layers[layer, channel, 0]
            elif x >= x_axis[channel, K - 1]:
                for layer in range(3):
                    out[layer, i, j] = y_layers[layer, channel, K - 1]
            else:
                low = np.searchsorted(x_axis[channel], x, side='right') - 1
                t = (x - x_axis[channel, low]) * inv_dx[channel, low]
                for layer in range(3):
                    y_layer = y_layers[layer, channel]
                    out[layer, i, j] = y_layer[low] + t * (y_layer[low + 1] - y_layer[low])
    return out


class DensityLayers:
    """Per-layer densities (x, y, layer, rgb) of ``density_cmy``, interpolated
    one (layer, channel) plane at a time so the full 4D array is never held.
    """

    def __init__(self, density_cmy, density_curves, density_curves_layers, positive_film=False):
        self._density_cmy = as_float_array(density_cmy)
        self.dtype = self._density_cmy.dtype
        self._sign = -1.0 if positive_film else 1.0
        self._x_axes = np.ascontiguousarray((self._sign * np.asarray(density_curves, dtype=np.float64)).T)
        dx = np.diff(self._x_axes, axis=1)
        self._inv_dx = np.zeros_like(dx)
        np.divide(1.0, dx, out=self._inv_dx, where=dx != 0)
        self._y_layers = np.ascontiguousarray(np.transpose(density_curves_layers, (1, 2, 0)), dtype=np.float64)
        self.shape = self._density_cmy.shape[0:2] + (3, 3)

    def layer(self, layer, channel):
        out = np.empty(self.shape[0:2], dtype=self.dtype)
        _interp_density_layer_kernel(self._density_cmy[:, :, channel], self._x_axes[channel],
                                     self._inv_dx[channel], self._y_layers[layer, channel],
                                     self._sign, out)
        return out

    def channel_layers(self, channel):
        """Return the three density sublayers of one colour channel."""
        return _interp_density_channel_layers_kernel(
            self._density_cmy, self._x_axes, self._inv_dx,
            self._y_layers, self._sign, channel,
        )


def interp_density_cmy_layers(density_cmy, density_curves, density_curves_layers, positive_film=False):
    """Split total CMY density into per-layer densities, shape (x, y, layer, rgb)."""
    layers = DensityLayers(density_cmy, density_curves, density_curves_layers, positive_film)
    density_cmy_layers = np.empty(layers.shape)
    for ch in range(3):
        for sl in range(3):
            density_cmy_layers[:, :, sl, ch] = layers.layer(sl, ch)
    return density_cmy_layers

# This method was used for multilayer grain, but it is not used anymore
# def interpolate_layers(self, exposure_rgb):
#     density_curves_layers = density_curves_layers_model(self.log_exposure, self.parameters, self.type)
#     density_cmy_layers = np.zeros((exposure_rgb.shape[0], exposure_rgb.shape[1], 3, 3))
#     exposure = 10**(self.log_exposure)
#     for channel in np.arange(3):
#         for layer in np.arange(3):
#             density_cmy_layers[:,:,channel,layer] = np.interp(exposure_rgb[:,:,channel],
#                                                               exposure,
#                                                               density_curves_layers[:,channel,layer])
#     return density_cmy_layers

def apply_gamma_shift_correction(log_exposure, density_curves, gamma_correction, log_exposure_correction):
    dc = density_curves
    le = log_exposure
    gc = gamma_correction
    les = log_exposure_correction
    dc_out = np.zeros_like(dc)
    for i in np.arange(3):
        dc_out[:,i] = np.interp(le, le/gc[i] + les[i], dc[:,i])
    return dc_out


def remove_viewing_glare_comp(le, dc, factor=0.2, density=1.0, transition=0.3):
    """
    Removes viewing glare compensation from the density curves of print paper.
    Parameters:
    le (numpy.ndarray): The log exposure values.
    dc (numpy.ndarray): density curves of the print paper. Shape (n,3).
    factor (float, optional): The factor by which to reduce the light exposure values of the shadows. (brighter shadows). Default is 0.1.
    density (float, optional): The density value of the transition point. Default is 1.2.
    transition (float, optional): The transition density range used for Gaussian filtering. Default is 0.3.
    Returns:
    numpy.ndarray: density curves with viewing glare compensation removed.
    """
    def _measure_slope(le, density_curve, le_center, range_ev=1):
        le_delta = np.log10(2**range_ev)/2
        le_0 = le_center - le_delta
        le_1 = le_center + le_delta
        density_0 = np.interp(le_0, le, density_curve)
        density_1 = np.interp(le_1, le, density_curve)
        slope = (density_1 - density_0)/(le_1 - le_0)
        return slope    
    
    dc_mean = np.mean(dc, axis=1)
    le_center = np.interp(density, dc_mean, le)
    slope = _measure_slope(le, dc_mean, le_center)
    le_step = np.mean(np.diff(le))
    dc_out = np.zeros_like(dc)
    for i in np.arange(3):
        le_nl = np.copy(le)
        le_nl[le>le_center] -= (le[le>le_center]-le_center)*factor
        le_transition = transition/slope
        le_nl = scipy.ndimage.gaussian_filter(le_nl, le_transition/le_step)
        dc_out[:,i] = np.interp(le_nl, le, dc[:,i])
    return dc_out
