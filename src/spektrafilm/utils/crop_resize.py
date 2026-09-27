import numpy as np
import skimage.transform

def crop_image(image, center=(0.5,0.5), size=(0.1, 0.1)):
    """
    Crop an image based on a specified fraction and center.

    Parameters:
    image (numpy.ndarray): The input image to be cropped.
    center (tuple of float, optional): The center of the cropping area as a tuple of two floats (x, y). 
                                      Each value should be between 0 and 1. Default is (0.5, 0.5).
    size (tuple of float, optional): The normalize size of the cropped area as fraction of the long side, (x,y). Default is (0.1, 0.1).

    Returns:
    numpy.ndarray: The cropped image.
    """
    center = np.flip(center)
    shape = image.shape[0:2]
    cn = np.round(shape*np.array(center))
    sz = np.round(np.double(np.max(shape))*np.flip(np.array(size)))
    x0 = np.round(cn - sz/2)
    sz = np.int64(sz)
    x0 = np.int64(x0)
    x0[x0<0] = 0
    if x0[0]+sz[0]>shape[0]: x0[0] = shape[0]-sz[0]
    if x0[1]+sz[1]>shape[1]: x0[1] = shape[1]-sz[1]
    image_crop = image[x0[0]:x0[0]+sz[0], x0[1]:x0[1]+sz[1],:]
    return image_crop

# def resize_image(image, resize_factor=1.0): #TBD
#     """
#     Resize the given image by a specified factor.
#     Parameters:
#     image (numpy.ndarray): The image to be resized.
#     resize_factor (float, optional): The factor by which to resize the image. 
#                                      Default is 1.0 (no resizing).
#     Returns:
#     numpy.ndarray: The resized image.
#     """
#     # zoom(image, zoom=(resize_factor, resize_factor, 1.0))
#     return skimage.transform.rescale(image, resize_factor, channel_axis=2)

def _mirror_index(idx, n):
    # scipy.ndimage mode='mirror': (d c b | a b c d | c b a)
    if n == 1:
        return np.zeros_like(idx)
    period = 2 * (n - 1)
    idx = np.abs(idx) % period
    return np.where(idx >= n, period - idx, idx)


def _sampled_gaussian_1d(image, axis, sigma, sample_idx):
    # scipy gaussian_filter1d (truncate=4, mode='mirror') evaluated only at sample_idx along axis
    if sigma <= 1e-15:  # gaussian_filter skips such axes
        return np.take(image, sample_idx, axis=axis)
    radius = int(4.0 * sigma + 0.5)
    offsets = np.arange(-radius, radius + 1)
    weights = np.exp(-0.5 / sigma**2 * offsets**2)
    weights /= weights.sum()
    n = image.shape[axis]
    out = None
    for weight, offset in zip(weights, offsets):
        term = weight * np.take(image, _mirror_index(sample_idx + offset, n), axis=axis)
        out = term if out is None else out + term
    return out


def rescale_nearest_antialiased(image, scale):
    """Same result as ``skimage.transform.rescale(image, scale, order=0,
    channel_axis=2)`` for a float image being downscaled, but the
    anti-aliasing Gaussian is evaluated only at the pixels the
    nearest-neighbour resampling keeps instead of over the whole image.
    """
    import scipy.ndimage as ndi

    image = np.asarray(image)
    in_shape = np.asarray(image.shape[:2])
    out_shape = np.maximum(np.round(scale * in_shape), 1).astype(int)
    factors = in_shape / out_shape
    sigmas = np.maximum(0, (factors - 1) / 2)
    out = image
    for axis in range(2):
        # indices kept by skimage's ndi.zoom(order=0, grid_mode=True) along this axis
        sample_idx = ndi.zoom(np.arange(in_shape[axis], dtype=np.float64), 1 / factors[axis],
                              order=0, mode='mirror', grid_mode=True).astype(np.intp)
        assert sample_idx.shape[0] == out_shape[axis]
        out = _sampled_gaussian_1d(out, axis, sigmas[axis], sample_idx)
    # skimage clips the output to the input range
    min_val = np.min(image)
    if np.isnan(min_val):
        min_val, max_val = np.nanmin(image), np.nanmax(image)
    else:
        max_val = np.max(image)
    return np.clip(out, min_val, max_val)
