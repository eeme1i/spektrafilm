import numpy as np
import scipy
import scipy.ndimage
from numba import njit, prange
from spektrafilm.model.density_curves import DensityLayers
from spektrafilm.runtime.params_schema import GrainParams
from spektrafilm.utils.fast_stats import binomial_sample, fast_lognormal_from_mean_std, poisson_sample
from spektrafilm.utils.fast_gaussian_filter import fast_gaussian_filter
from spektrafilm.utils import gpu_grain

################################################################################
# Grain (very simple model)
################################################################################

def layer_particle_model(density,
                         density_max=2.2,
                         n_particles_per_pixel=10,
                         grain_uniformity=0.98,
                         seed=None,
                         blur_particle=0.0,
                         method='poisson_binomial',
                         use_fast_stats=False,
                         use_gpu=False,
                         ):
    if seed is not None:
        np.random.seed(seed) # scipy uses np.random
    
    od_particle = density_max/n_particles_per_pixel
    if method=='poisson_binomial' and use_fast_stats:
        grain = None
        if use_gpu:
            # seeded from np.random, so per-layer seeds carry over to the GPU
            grain = gpu_grain.poisson_binomial_grain(density, float(density_max),
                                                     float(n_particles_per_pixel), float(grain_uniformity))
        if grain is None:
            grain = _fast_poisson_binomial_grain(density, float(density_max),
                                                 float(n_particles_per_pixel), float(grain_uniformity))
        grain = grain.astype(np.asarray(density).dtype, copy=False)
        if blur_particle>0:
            grain = fast_gaussian_filter(grain, blur_particle*np.sqrt(od_particle))
        return grain

    probability_of_development = density/density_max
    probability_of_development = np.clip(probability_of_development, 1e-6, 1-1e-6) # for safe calc
    
    grain = np.zeros_like(density)
    if method=='gamma_beta':
        gamma_rvs = scipy.stats.gamma.rvs
        beta_rvs = scipy.stats.beta.rvs
        seeds = gamma_rvs(n_particles_per_pixel/(1-grain_uniformity+1e-6), size=density.shape) * (1-grain_uniformity+1e-6)
        grain = beta_rvs(probability_of_development*n_particles_per_pixel,
                        (1-probability_of_development)*n_particles_per_pixel)*seeds*od_particle
    elif method=='poisson_binomial':
        binom_rvs = scipy.stats.binom.rvs
        poisson_rvs = scipy.stats.poisson.rvs
        saturation = 1 - probability_of_development*grain_uniformity*(1-1e-6)
        seeds = poisson_rvs(n_particles_per_pixel/saturation)
        grain = binom_rvs(seeds, probability_of_development)
        grain = np.double(grain)*od_particle*saturation
    
    if blur_particle>0:
        # grain = scipy.ndimage.gaussian_filter(grain, blur_particle*np.sqrt(od_particle))
        grain = fast_gaussian_filter(grain, blur_particle*np.sqrt(od_particle))
    return grain

@njit(parallel=True, cache=True)
def _fast_poisson_binomial_grain(density, density_max, n_particles_per_pixel, grain_uniformity):
    # Fused per-pixel version of the poisson_binomial branch of layer_particle_model
    # using the fast_stats samplers (same arithmetic, one pass, no temporaries).
    n_rows, n_cols = density.shape
    od_particle = density_max/n_particles_per_pixel
    grain = np.empty((n_rows, n_cols), dtype=density.dtype)
    for i in prange(n_rows):
        for j in range(n_cols):
            probability = min(max(density[i, j]/density_max, 1e-6), 1-1e-6)
            saturation = 1 - probability*grain_uniformity*(1-1e-6)
            seeds = poisson_sample(n_particles_per_pixel/saturation)
            grain[i, j] = binomial_sample(seeds, probability)*od_particle*saturation
    return grain

def add_micro_structure(density_cmy_out, micro_structure, pixel_size_um):
    grain_micro_structure_blur_pixel = micro_structure[0]/pixel_size_um
    grain_micro_structure_sigma = micro_structure[1]*0.001/pixel_size_um  # grain microstructure[1] is in nm
    if grain_micro_structure_sigma > 0.05:
        clumping = fast_lognormal_from_mean_std(np.ones_like(density_cmy_out),
                                                np.full_like(density_cmy_out, grain_micro_structure_sigma))
        if grain_micro_structure_blur_pixel>0.4:
            # clumping = scipy.ndimage.gaussian_filter(clumping, (grain_micro_structure_blur_pixel,
            #                                                     grain_micro_structure_blur_pixel, 0))
            clumping = fast_gaussian_filter(clumping, grain_micro_structure_blur_pixel)
        density_cmy_out *= clumping
    return density_cmy_out

def apply_grain_to_density(density_cmy,
                           pixel_size_um=10,
                           particle_area_um2=0.2,
                           particle_scale=[1,0.8,3],
                           density_min=[0.03,0.06,0.04],
                           density_max_curves=[2.2,2.2,2.2],
                           grain_uniformity=[0.98,0.98,0.98],
                           grain_blur=1.0,
                           n_sub_layers=1,
                           fixed_seed=None,
                           ):
    density_min = np.array(density_min)
    density_max = density_max_curves + density_min
    pixel_area_um2 = pixel_size_um**2
    particle_area_um2 = particle_area_um2*np.array(particle_scale)
    n_particles_per_pixel = pixel_area_um2/particle_area_um2
    sigma_blur_pixel = grain_blur
    
    if fixed_seed is not None:
        seed = None
    else:
        seed = [0, 1, 2]
    
    if n_sub_layers>1:
        n_particles_per_pixel /= n_sub_layers
    
    density_cmy += density_min
    density_cmy_out = np.zeros_like(density_cmy)
    for ch in np.arange(3):
        for sl in np.arange(n_sub_layers):
            density_cmy_out[:,:,ch] += layer_particle_model(density_cmy[:,:,ch],
                                                            density_max=density_max[ch],
                                                            n_particles_per_pixel=n_particles_per_pixel[ch],
                                                            grain_uniformity=grain_uniformity[ch],
                                                            seed=seed[ch] + sl*10)
    density_cmy_out /= n_sub_layers
    density_cmy_out -= density_min
    
    if sigma_blur_pixel>0.4:
        # density_cmy_out = scipy.ndimage.gaussian_filter(density_cmy_out, (sigma_blur_pixel, sigma_blur_pixel, 0))
        density_cmy_out = fast_gaussian_filter(density_cmy_out, sigma_blur_pixel)
        
    return density_cmy_out


# experimental
def apply_grain_to_density_layers(density_cmy_layers, # x,y,sublayers,rgb array, or DensityLayers
                                  density_max_layers, # 3x3 [sublayers,rgb]
                                  pixel_size_um=10,
                                  particle_area_um2=0.2,
                                  particle_scale=[1,0.8,3], # rgb
                                  particle_scale_layers=[3,1,0.3], # sublayers
                                  density_min=[0.03,0.06,0.04],
                                  grain_uniformity=[0.98,0.98,0.98],
                                  grain_blur=1.0,
                                  grain_blur_dye_clouds_um=1.0,
                                  grain_micro_structure=(0.1, 30),
                                  fixed_seed=None,
                                  use_fast_stats=False,
                                  use_gpu=False,
                                  ):
    density_max_total = np.sum(density_max_layers, axis=0) # [sublayers,rgb]
    density_max_fractions = density_max_layers/density_max_total[None,:]
    density_min_layers = density_max_fractions*np.array(density_min)[None,:]
    density_max_layers = density_max_layers + density_min_layers
    
    pixel_area_um2 = pixel_size_um**2
    particle_area_um2_layers = (particle_area_um2 * 
                                    np.array(particle_scale)[None,:] * 
                                    np.array(particle_scale_layers)[:,None]) # layers, rgb
    n_particles_per_pixel = pixel_area_um2*density_max_fractions/particle_area_um2_layers

    
    if fixed_seed is not None:
        seed = None
    else:
        seed = [0, 1, 2]
    
    density_cmy_out = np.zeros(density_cmy_layers.shape[0:3], dtype=density_cmy_layers.dtype)
    for ch in np.arange(3): # rgb channels
        channel_layers = (
            density_cmy_layers.channel_layers(ch)
            if isinstance(density_cmy_layers, DensityLayers) else None
        )
        for sl in np.arange(3): # sublayers
            if channel_layers is not None:
                density_layer = channel_layers[sl]
            else:
                density_layer = density_cmy_layers[:,:,sl,ch]
            density_cmy_out[:,:,ch] += layer_particle_model(density_layer + float(density_min_layers[sl,ch]),
                                                            density_max=density_max_layers[sl,ch],
                                                            n_particles_per_pixel=n_particles_per_pixel[sl,ch],
                                                            grain_uniformity=grain_uniformity[ch],
                                                            seed=seed[ch] + sl*10,
                                                            blur_particle=grain_blur_dye_clouds_um,
                                                            use_fast_stats=use_fast_stats,
                                                            use_gpu=use_gpu)
    
    # micro-structure
    density_cmy_out = add_micro_structure(density_cmy_out, grain_micro_structure, pixel_size_um)

    # final
    density_cmy_out -= density_min
    if grain_blur>0:
        # density_cmy_out = scipy.ndimage.gaussian_filter(density_cmy_out, (grain_blur, grain_blur, 0))
        density_cmy_out = fast_gaussian_filter(density_cmy_out, grain_blur)
    return density_cmy_out


def apply_grain(
    density_cmy,
    pixel_size_um,
    grain: GrainParams,
    density_curves,
    density_curves_layers,
    profile_type,
    bypass_grain=False,
    use_fast_stats=False,
    use_gpu=False,
):
    if not grain.active or bypass_grain:
        return density_cmy

    if not grain.sublayers_active:
        density_max = np.nanmax(density_curves, axis=0)
        return apply_grain_to_density(
            density_cmy,
            pixel_size_um=pixel_size_um,
            particle_area_um2=grain.particle_area_um2,
            particle_scale=grain.particle_scale,
            density_min=grain.density_min,
            density_max_curves=density_max,
            grain_uniformity=grain.uniformity,
            grain_blur=grain.blur,
            n_sub_layers=grain.n_sub_layers,
        )

    density_cmy_layers = DensityLayers(
        density_cmy,
        density_curves,
        density_curves_layers,
        positive_film=profile_type == 'positive',
    )
    density_max_layers = np.nanmax(density_curves_layers, axis=0)
    return apply_grain_to_density_layers(
        density_cmy_layers,
        density_max_layers=density_max_layers,
        pixel_size_um=pixel_size_um,
        particle_area_um2=grain.particle_area_um2,
        particle_scale=grain.particle_scale,
        particle_scale_layers=grain.particle_scale_layers,
        density_min=grain.density_min,
        grain_uniformity=grain.uniformity,
        grain_blur=grain.blur,
        grain_blur_dye_clouds_um=grain.blur_dye_clouds_um,
        grain_micro_structure=grain.micro_structure,
        use_fast_stats=use_fast_stats,
        use_gpu=use_gpu,
    )

# TODO: make grain parameter with RMS granularity

if __name__=='__main__':
    density = np.ones((128,128))*2
    g1 = layer_particle_model(density, density_max=2, n_particles_per_pixel=10, grain_uniformity=0.99, sigma_blur=0.)
    g2 = layer_particle_model(density, density_max=2, n_particles_per_pixel=10, grain_uniformity=0.96, sigma_blur=0.)
    print('g1 ------------------')
    print('Density Test')
    print('Mean', np.mean(g1))
    print('RMS', np.std(g1)*1000)
    print('Skewness', scipy.stats.skew(g1.flatten()))
    print('Kurtosis', scipy.stats.kurtosis(g1.flatten()))
    print('g2 ------------------')
    print('Mean', np.mean(g2))
    print('RMS', np.std(g2)*1000)
    print('Skewness', scipy.stats.skew(g2.flatten()))
    print('Kurtosis', scipy.stats.kurtosis(g2.flatten()))
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(g1, vmin=0, vmax=2.2)
    axs[0].set_title('Uniformity=0.99')
    axs[1].imshow(g2, vmin=0, vmax=2.2)
    axs[1].set_title('Uniformity=0.96')
    fig.suptitle('Fully saturated density with different uniformity')
    plt.show()
