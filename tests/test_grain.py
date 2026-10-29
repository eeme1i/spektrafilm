import numpy as np
import pytest

from spektrafilm.model.density_curves import DensityLayers, interp_density_cmy_layers
from spektrafilm.model.grain import apply_grain_to_density, apply_grain_to_density_layers
from spektrafilm.model.grain import apply_grain
from spektrafilm.runtime.params_schema import GrainParams
from spektrafilm.profiles.io import list_profiles, load_profile


pytestmark = pytest.mark.unit


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('stock', [
    stock for stock in list_profiles() if load_profile(stock).is_filming
])
def test_bundled_grain_splits_preserve_parent_density(stock, dtype):
    profile = load_profile(stock)
    curves = profile.data.density_curves
    curves = curves - curves.min(axis=0)
    maximum = curves.max(axis=0)
    # Include parent knots, intermediate values, and both clamped endpoints.
    ramp = np.linspace(0, 1, 513)[:, None] * maximum
    density = np.concatenate([curves, ramp, [-maximum], [maximum * 2]])[None].astype(dtype)
    layers = DensityLayers(density, curves, profile.data.density_curves_layers, profile.is_positive)
    split = np.stack([layers.channel_layers(c) for c in range(3)], axis=-1)
    expected = np.clip(density, 0, maximum)
    tolerance = 8 * np.finfo(dtype).eps
    np.testing.assert_allclose(split.sum(axis=0), expected, rtol=tolerance, atol=tolerance)
    assert np.all(split >= 0)
    assert np.all(np.isfinite(split))
    assert split.dtype == dtype
    # Sampling capacities come from the same corrected table as the split.
    knot_layers = DensityLayers(curves[None], curves, profile.data.density_curves_layers, profile.is_positive)
    at_knots = np.stack([knot_layers.channel_layers(c) for c in range(3)], axis=-1)
    np.testing.assert_allclose(at_knots.max(axis=(1, 2)), layers.density_max_layers, atol=1e-12)


def test_grain_split_preserves_fitted_proportions_at_knots():
    curves = np.tile(np.array([0.0, 0.5, 1.0])[:, None], (1, 3))
    fitted = np.tile(np.array([[0.02, 0.01, 0.03], [0.3, 0.1, 0.2], [0.1, 0.4, 0.7]])[:, :, None], (1, 1, 3))
    layers = interp_density_cmy_layers(curves[None], curves, fitted)
    np.testing.assert_array_equal(layers[0, 0], 0)
    np.testing.assert_allclose(layers[0, 1:] / curves[1:, None, :],
                               fitted[1:] / fitted[1:].sum(axis=1)[:, None, :])


def test_grain_split_rejects_missing_proportions_for_positive_density():
    curves = np.tile(np.array([0.0, 1.0])[:, None], (1, 3))
    with pytest.raises(ValueError, match='nonzero fitted sublayer'):
        DensityLayers(curves[None], curves, np.zeros((2, 3, 3)))


@pytest.mark.parametrize('stock', ['kodak_portra_400', 'fujifilm_pro_400h', 'fujifilm_provia_100f'])
def test_layered_grain_without_sampling_noise_preserves_density(monkeypatch, stock):
    # Identity sampling isolates deterministic changes introduced by enabling grain.
    monkeypatch.setattr('spektrafilm.model.grain.layer_particle_model',
                        lambda density, **kwargs: density)
    profile = load_profile(stock)
    curves = profile.data.density_curves
    curves = curves - curves.min(axis=0)
    density = (np.linspace(0, 1, 257)[None, :, None] * curves.max(axis=0)).astype(np.float32)
    grain = GrainParams(blur=0, blur_dye_clouds_um=0, micro_structure=(0, 0))
    result = apply_grain(density.copy(), 4.0, grain, curves,
                         profile.data.density_curves_layers, profile.info.type)
    np.testing.assert_allclose(result, density, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('positive_film', [False, True])
def test_channel_layers_matches_individual_interpolation_exactly(dtype, positive_film):
    rng = np.random.default_rng(29)
    exposure = np.linspace(0.0, 2.5, 35)
    curves = np.stack([exposure, exposure * 0.9, exposure * 0.8], axis=-1)
    curve_layers = np.stack([curves * 0.5, curves * 0.3, curves * 0.2], axis=1)
    density = rng.uniform(-0.5, 3.0, size=(32, 47, 3)).astype(dtype)
    layers = DensityLayers(density, curves, curve_layers, positive_film=positive_film)

    for channel in range(3):
        shared = layers.channel_layers(channel)
        for layer in range(3):
            np.testing.assert_array_equal(shared[layer], layers.layer(layer, channel))


class TestApplyGrain:
    def test_apply_grain_returns_input_when_bypassed_or_inactive(self):
        density_cmy = np.full((3, 3, 3), 0.4)
        density_curves = np.tile(np.linspace(0.0, 2.0, 8)[:, None], (1, 3))
        density_curves_layers = np.tile(density_curves[:, None, :] / 3.0, (1, 3, 1))
        grain = GrainParams(active=False)

        inactive = apply_grain(
            density_cmy.copy(),
            4.0,
            grain,
            density_curves,
            density_curves_layers,
            "negative",
        )
        bypassed = apply_grain(
            density_cmy.copy(),
            4.0,
            GrainParams(active=True),
            density_curves,
            density_curves_layers,
            "negative",
            bypass_grain=True,
        )

        np.testing.assert_allclose(inactive, density_cmy, atol=1e-10)
        np.testing.assert_allclose(bypassed, density_cmy, atol=1e-10)

    def test_apply_grain_matches_single_layer_pipeline(self):
        density_cmy = np.full((4, 4, 3), [0.3, 0.6, 0.9], dtype=np.float64)
        density_curves = np.column_stack([
            np.linspace(0.0, 2.4, 12),
            np.linspace(0.0, 2.2, 12),
            np.linspace(0.0, 2.0, 12),
        ])
        density_curves_layers = np.tile(density_curves[:, None, :] / 3.0, (1, 3, 1))
        grain = GrainParams(
            active=True,
            sublayers_active=False,
            particle_area_um2=0.25,
            particle_scale=(0.9, 1.1, 1.4),
            density_min=(0.05, 0.07, 0.09),
            uniformity=(0.98, 0.97, 0.96),
            blur=0.0,
            n_sub_layers=2,
        )

        result = apply_grain(
            density_cmy.copy(),
            5.0,
            grain,
            density_curves,
            density_curves_layers,
            "negative",
        )

        expected = apply_grain_to_density(
            density_cmy.copy(),
            pixel_size_um=5.0,
            particle_area_um2=grain.particle_area_um2,
            particle_scale=grain.particle_scale,
            density_min=grain.density_min,
            density_max_curves=np.nanmax(density_curves, axis=0),
            grain_uniformity=grain.uniformity,
            grain_blur=grain.blur,
            n_sub_layers=grain.n_sub_layers,
        )

        np.testing.assert_allclose(result, expected, atol=1e-10)

    @pytest.mark.parametrize("profile_type", ["negative", "positive"])
    def test_apply_grain_matches_layered_pipeline(self, profile_type):
        density_cmy = np.full((4, 4, 3), [0.35, 0.55, 0.75], dtype=np.float64)
        density_curves = np.column_stack([
            np.linspace(0.0, 2.1, 10),
            np.linspace(0.0, 1.9, 10),
            np.linspace(0.0, 1.7, 10),
        ])
        density_curves_layers = np.stack([
            density_curves * np.array([0.55, 0.50, 0.45]),
            density_curves * np.array([0.30, 0.33, 0.35]),
            density_curves * np.array([0.15, 0.17, 0.20]),
        ], axis=1)
        grain = GrainParams(
            active=True,
            sublayers_active=True,
            particle_area_um2=0.18,
            particle_scale=(0.8, 1.0, 1.2),
            particle_scale_layers=(2.2, 1.0, 0.5),
            density_min=(0.04, 0.06, 0.08),
            uniformity=(0.99, 0.98, 0.97),
            blur=0.0,
            blur_dye_clouds_um=0.0,
            micro_structure=(0.0, 0.0),
        )

        result = apply_grain(
            density_cmy.copy(),
            4.0,
            grain,
            density_curves,
            density_curves_layers,
            profile_type,
            use_fast_stats=False,
        )

        density_cmy_layers = interp_density_cmy_layers(
            density_cmy.copy(),
            density_curves,
            density_curves_layers,
            positive_film=profile_type == "positive",
        )
        expected = apply_grain_to_density_layers(
            density_cmy_layers,
            density_max_layers=np.nanmax(density_curves_layers, axis=0),
            pixel_size_um=4.0,
            particle_area_um2=grain.particle_area_um2,
            particle_scale=grain.particle_scale,
            particle_scale_layers=grain.particle_scale_layers,
            density_min=grain.density_min,
            grain_uniformity=grain.uniformity,
            grain_blur=grain.blur,
            grain_blur_dye_clouds_um=grain.blur_dye_clouds_um,
            grain_micro_structure=grain.micro_structure,
            use_fast_stats=False,
        )

        np.testing.assert_allclose(result, expected, atol=1e-10)


@pytest.mark.parametrize('sublayers', [False, True])
@pytest.mark.parametrize('use_fast_stats', [False, True])
def test_explicit_grain_seed_repeats_and_changes_without_global_rng(sublayers, use_fast_stats):
    profile = load_profile('kodak_portra_400')
    curves = profile.data.density_curves
    curves = curves - curves.min(axis=0)
    density = np.full((32, 48, 3), 0.7, dtype=np.float32)
    # Exercise clumping as well as particle sampling; ordinary preview pixel
    # sizes would make microstructure too small to activate this branch.
    grain = GrainParams(seed=124, sublayers_active=sublayers, blur=0,
                        blur_dye_clouds_um=0, micro_structure=(0, 100))
    state = np.random.get_state()

    def sample(seed):
        grain.seed = seed
        return apply_grain(density.copy(), 1.0, grain, curves,
                           profile.data.density_curves_layers, 'negative',
                           use_fast_stats=use_fast_stats, use_gpu=True)

    first = sample(124)
    other = sample(125)
    repeated = sample(124)
    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, other)
    assert first.dtype == density.dtype
    assert np.isfinite(first).all()
    assert first.std() > 0
    after = np.random.get_state()
    assert state[0] == after[0]
    np.testing.assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]


def test_simulator_frame_seed_is_repeatable_and_restored_after_error():
    from spektrafilm import Simulator, digest_params, init_params
    params = init_params('kodak_portra_400', 'kodak_portra_endura')
    params.camera.auto_exposure = False
    params.film_render.grain.seed = 7
    params.print_render.glare.active = False
    params.film_render.glare.active = False
    params.settings.use_enlarger_lut = True
    params.settings.use_scanner_lut = True
    simulator = Simulator(digest_params(params))
    image = np.full((16, 24, 3), 0.184, dtype=np.float32)
    first = simulator.process(image.copy(), grain_seed=12)
    other = simulator.process(image.copy(), grain_seed=13)
    repeated = simulator.process(image.copy(), grain_seed=12)
    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, other)
    # A failing frame must not leave its seed installed for the next call.
    with pytest.raises(ValueError):
        simulator.process(image.copy(), grain_seed=-1)
    np.testing.assert_array_equal(simulator.process(image.copy()),
                                  simulator.process(image.copy(), grain_seed=7))
