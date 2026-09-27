from __future__ import annotations

import numpy as np
import pytest

from spektrafilm.model.grain import _fast_poisson_binomial_grain, layer_particle_model
from spektrafilm.utils import gpu_grain


pytestmark = pytest.mark.unit

DENSITY_MAX = 2.2
UNIFORMITY = 0.98


def _cpu_sampler(density, n_particles):
    return _fast_poisson_binomial_grain(density, DENSITY_MAX, n_particles, UNIFORMITY)


def _gpu_sampler(density, n_particles):
    return gpu_grain.poisson_binomial_grain(density, DENSITY_MAX, n_particles, UNIFORMITY)


SAMPLERS = [
    pytest.param(_cpu_sampler, id='cpu'),
    pytest.param(_gpu_sampler, id='gpu', marks=pytest.mark.skipif(
        not gpu_grain.available(), reason='MLX / Metal not available')),
]


def _expected_moments(density, n_particles):
    # Binomial thinning of a Poisson count is Poisson(n_particles * p / saturation),
    # scaled by od_particle * saturation.
    p = np.clip(density / DENSITY_MAX, 1e-6, 1 - 1e-6)
    saturation = 1 - p * UNIFORMITY * (1 - 1e-6)
    od_particle = DENSITY_MAX / n_particles
    return DENSITY_MAX * p, od_particle**2 * saturation * n_particles * p


# n_particles spans every sampler branch: Knuth and normal-approx Poisson,
# direct Bernoulli, inversion and normal-approx binomial.
@pytest.mark.parametrize('sampler', SAMPLERS)
@pytest.mark.parametrize('n_particles', [5.0, 20.0, 80.0, 400.0])
@pytest.mark.parametrize('density', [0.1, 0.8, 1.6, 2.15])
def test_grain_sampler_moments(sampler, n_particles, density) -> None:
    field = np.full((400, 500), density, dtype=np.float32)

    grain = sampler(field, n_particles).astype(np.float64)

    mean, var = _expected_moments(density, n_particles)
    assert grain.mean() == pytest.approx(mean, rel=0.01)
    assert grain.var() == pytest.approx(var, rel=0.04)


@pytest.mark.parametrize('sampler', SAMPLERS)
def test_grain_sampler_is_spatially_uncorrelated(sampler) -> None:
    grain = sampler(np.full((500, 500), 1.0, dtype=np.float32), 20.0).astype(np.float64)
    grain -= grain.mean()

    for shifted, reference in ((grain[1:, :], grain[:-1, :]), (grain[:, 1:], grain[:, :-1])):
        assert abs(np.corrcoef(shifted.ravel(), reference.ravel())[0, 1]) < 0.01


@pytest.mark.skipif(not gpu_grain.available(), reason='MLX / Metal not available')
def test_gpu_layers_get_independent_noise() -> None:
    density = np.full((300, 300), 1.0, dtype=np.float32)
    kwargs = dict(density_max=DENSITY_MAX, n_particles_per_pixel=20.0,
                  grain_uniformity=UNIFORMITY, use_fast_stats=True, use_gpu=True)

    layer_a = layer_particle_model(density, seed=0, **kwargs)
    layer_b = layer_particle_model(density, seed=10, **kwargs)
    layer_a_again = layer_particle_model(density, seed=0, **kwargs)

    assert layer_a.dtype == np.float32
    assert abs(np.corrcoef(layer_a.ravel(), layer_b.ravel())[0, 1]) < 0.02
    np.testing.assert_array_equal(layer_a, layer_a_again)


def test_gpu_path_falls_back_to_cpu_when_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(gpu_grain, 'poisson_binomial_grain', lambda *args, **kwargs: None)
    density = np.full((64, 64), 1.0, dtype=np.float32)

    grain = layer_particle_model(density, density_max=DENSITY_MAX, n_particles_per_pixel=20.0,
                                 grain_uniformity=UNIFORMITY, use_fast_stats=True, use_gpu=True)

    assert grain.shape == density.shape
    assert grain.mean() == pytest.approx(1.0, rel=0.05)
