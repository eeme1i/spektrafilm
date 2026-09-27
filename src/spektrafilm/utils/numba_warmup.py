import numpy as np

from spektrafilm.utils.fast_stats import warmup_fast_stats
from spektrafilm.utils.lut import warmup_luts
from spektrafilm.utils.fast_interp import warmup_fast_interp
from spektrafilm.utils.fast_gaussian_filter import warmup_fast_gaussian_filter
from spektrafilm.utils.numba_boost_hightlights import warmup_boost_highlights
from spektrafilm.utils.fast_cam16ucs import warmup_fast_cam16ucs
from spektrafilm.utils.conversions import warmup_conversions
from spektrafilm.utils.fast_colour import warmup_fast_colour
from spektrafilm.utils.gpu_grain import warmup_gpu_grain

# precompile numba functions
def warmup():
    warmup_fast_stats()
    warmup_luts()
    warmup_fast_interp()
    warmup_fast_gaussian_filter()
    warmup_boost_highlights()
    warmup_fast_cam16ucs()
    warmup_conversions()
    warmup_fast_colour()
    warmup_float32()
    warmup_gpu_grain()


def warmup_float32():
    """Compile the float32 specializations of the image-path kernels.

    The runtime pipeline carries images as float32 (see utils/dtypes.py),
    and numba compiles one specialization per dtype.
    """
    from spektrafilm.model.density_curves import DensityLayers
    from spektrafilm.model.grain import layer_particle_model
    from spektrafilm.utils.fast_cam16ucs import cam16ucs_to_xyz, xyz_to_cam16ucs
    from spektrafilm.utils.fast_gaussian_filter import fast_exponential_filter, fast_gaussian_filter
    from spektrafilm.utils.fast_interp import fast_interp
    from spektrafilm.utils.fast_interp_lut import apply_lut_3d, apply_lut_cubic_2d
    from spektrafilm.utils.fast_stats import fast_lognormal_from_mean_std
    from spektrafilm.utils.gamut_compression import _c_max_lookup
    from spektrafilm.utils.numba_boost_hightlights import boost_highlights

    rng = np.random.default_rng(0)
    image = rng.random((8, 8, 3)).astype(np.float32)
    plane = np.ascontiguousarray(image[:, :, 0])

    boost_highlights(image, boost_ev=1.0, boost_range=0.5, protect_ev=0.0)
    fast_gaussian_filter(image, 1.0)
    fast_gaussian_filter(image, 5.0)
    fast_gaussian_filter(plane, 1.0)
    fast_exponential_filter(image, np.array([3.0, 5.0, 7.0]))
    fast_lognormal_from_mean_std(np.ones_like(image), np.full_like(image, 0.1))

    grid = np.linspace(0.0, 1.0, 5)
    curves = np.repeat(grid[:, None], 3, axis=1)
    fast_interp(image, grid, curves)
    fast_interp(image, curves, curves)
    layers = DensityLayers(image, curves, np.repeat(curves[:, None, :], 3, axis=1))
    layer_particle_model(layers.layer(0, 0), density_max=2.2, n_particles_per_pixel=10.0,
                         grain_uniformity=0.98, blur_particle=1.0, use_fast_stats=True)

    lut_3d = np.stack(np.meshgrid(grid, grid, grid, indexing='ij'), axis=-1) ** 2
    apply_lut_3d(lut_3d, image)
    apply_lut_3d(lut_3d, image, method='mitchell')
    apply_lut_cubic_2d(rng.random((5, 5, 3)), np.ascontiguousarray(image[:, :, 0:2]))

    xyz_w = np.array([0.95047, 1.0, 1.08883])
    jab = xyz_to_cam16ucs(image, xyz_w, 64.0, 20.0)
    cam16ucs_to_xyz(jab, xyz_w, 64.0, 20.0)
    _c_max_lookup(plane, plane, grid, np.linspace(-np.pi, np.pi, 5), rng.random((5, 5)))
