"""Reproducible quality probes; writes JSON and never changes runtime defaults.

Run: uv run python scripts/probe_quality.py --out /tmp/quality-probe.json
Synthetic targets diagnose numerical/model behavior, not fidelity to real film.
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
from pathlib import Path
from time import perf_counter

import colour
import numpy as np
import numba
import scipy
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares

from spektrafilm import digest_params, init_params
from spektrafilm.model.couplers import compute_dir_couplers_matrix
from spektrafilm.model.density_curves import DensityLayers
from spektrafilm.model.grain import apply_grain, layer_particle_model
from spektrafilm.profiles.io import list_profiles, load_profile
from spektrafilm.runtime.pipeline import SimulationPipeline
from spektrafilm.utils.fast_gaussian_filter import (
    _EXPONENTIAL_GAUSSIAN_FITS, fast_gaussian_filter,
)
from spektrafilm.utils.lut import compute_with_lut


def errors(a, b):
    d = np.abs(np.asarray(a) - np.asarray(b))
    return {"mean_abs": float(d.mean()), "p99_abs": float(np.percentile(d, 99)),
            "max_abs": float(d.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("/tmp/quality-probe.json"))
    args = parser.parse_args()
    rng = np.random.default_rng(20261011)
    report = {"seed": 20261011,
              "environment": {"python": platform.python_version(), "platform": platform.platform(),
                              "numpy": np.__version__, "scipy": scipy.__version__,
                              "colour": colour.__version__, "numba": numba.__version__},
              "randomness": "Targets are seeded; fast parallel Numba grain draws are not reproducibly seeded.",
              "profiles": {}, "pipeline_luts": {},
              "grain_sampling": [], "grain_lut_bounds": [], "gaussian": [], "exponential_mtf": {}}

    # A valid split must preserve the parent density when no noise is added.
    for stock in list_profiles():
        p = load_profile(stock)
        if not p.is_film or not p.is_filming:
            continue
        dc = p.data.density_curves - np.nanmin(p.data.density_curves, axis=0)
        reconstructed = p.data.density_curves_layers.sum(axis=1)
        ramp = np.stack([np.linspace(0, dc[:, c].max(), 256) for c in range(3)], -1)[None]
        layers = DensityLayers(ramp, dc, p.data.density_curves_layers, p.is_positive)
        split = sum(np.stack([layers.layer(l, c) for c in range(3)], -1) for l in range(3))
        params = digest_params(init_params(stock))
        silver = dc.max(axis=0) - dc if p.is_positive else dc
        corrected_axis = p.data.log_exposure[:, None] - silver @ compute_dir_couplers_matrix(params.film_render.dir_couplers)
        report["profiles"][stock] = {
            "curve_split_error": errors(reconstructed, dc),
            "runtime_split_error": errors(split, ramp),
            "dir_corrected_axis_min_step": float(np.diff(corrected_axis, axis=0).min()),
            "sensitivity_shape": list(p.data.log_sensitivity.shape),
        }

    # Repeatable gamut/exposure stress target: 9^3 linear-sRGB patches at 4 levels.
    grid = np.stack(np.meshgrid(*([np.linspace(0, 1, 9)] * 3), indexing="ij"), -1).reshape(27, 27, 3)
    target = np.concatenate([grid * scale for scale in (0.03, 0.18, 1.0, 4.0)], axis=0).astype(np.float32)
    for stock in ("kodak_portra_400", "fujifilm_pro_400h", "fujifilm_xtra_400"):
        params = init_params(stock)
        params.io.input_color_space = "sRGB"
        params.io.input_cctf_decoding = False
        params.debug.lut_mode = True
        params = digest_params(params)
        exact_pipe = SimulationPipeline(params)
        exact = exact_pipe.process(target)
        xyz_exact = colour.sRGB_to_XYZ(exact)
        lab_exact = colour.XYZ_to_Lab(xyz_exact)
        in_gamut = np.all((exact >= 0) & (exact <= 1), axis=-1)
        measurements = {"target_pixels": int(in_gamut.size), "in_gamut_pixels": int(in_gamut.sum()), "resolutions": {}}
        # Isolate LUT clipping outside declared density bounds from interpolation error.
        upper = params.film.data.density_curves.max(axis=0)
        lower = -np.asarray(params.film_render.grain.density_min)
        density = rng.uniform(lower, upper, (64, 64, 3)).astype(np.float32)
        outside = np.broadcast_to(upper + 0.1, (4, 4, 3)).copy()
        fn = exact_pipe._printing_stage._film_cmy_to_print_log_raw
        outside_lut, _ = compute_with_lut(outside, fn, lower, upper, steps=17)
        measurements["enlarger_above_bound_0_1D_log_exposure_error"] = errors(outside_lut, fn(outside))
        for resolution in (17, 33, 65):
            fast_params = copy.deepcopy(params)
            fast_params.settings.use_enlarger_lut = True
            fast_params.settings.use_scanner_lut = True
            fast_params.settings.lut_resolution = resolution
            pipe = SimulationPipeline(fast_params)
            start = perf_counter()
            fast = pipe.process(target)
            cold = perf_counter() - start
            start = perf_counter()
            pipe.process(target)
            warm = perf_counter() - start
            delta = colour.delta_E(lab_exact, colour.XYZ_to_Lab(colour.sRGB_to_XYZ(fast)), method="CIE 2000")
            lut_out, _ = compute_with_lut(density, fn, lower, upper, steps=resolution)
            measurements["resolutions"][resolution] = {
                "rgb_error": errors(fast, exact),
                "delta_e_00_mean": float(delta.mean()),
                "delta_e_00_p99": float(np.percentile(delta, 99)),
                "delta_e_00_max": float(delta.max()),
                "in_gamut_delta_e_00_p99": float(np.percentile(delta[in_gamut], 99)),
                "enlarger_in_bounds_log_exposure_error": errors(lut_out, fn(density)),
                "cold_seconds": cold, "warm_seconds": warm,
            }
        report["pipeline_luts"][stock] = measurements
        print(f"Completed LUT probes: {stock}", flush=True)

        # Flat density patches isolate realistic grain excursions at 8064 px / 35 mm.
        for fraction in (0.0, 0.5, 0.95, 1.0):
            dc = params.film.data.density_curves
            normalized = dc - dc.min(axis=0)
            patch = np.broadcast_to(normalized.max(axis=0)*fraction, (256, 256, 3)).astype(np.float32).copy()
            # lut_mode disabled grain in params; take ordinary stock rendering defaults.
            grain = digest_params(init_params(stock)).film_render.grain
            sampled = apply_grain(patch, 35000/8064, grain, normalized,
                                  params.film.data.density_curves_layers, params.film.info.type,
                                  use_fast_stats=True, use_gpu=False)
            clipped = np.clip(sampled, lower, upper)
            subset = sampled[::4, ::4]
            report["grain_lut_bounds"].append({
                "stock": stock, "fraction_of_dmax": fraction, "pixel_size_um": 35000/8064,
                "any_channel_outside_fraction": float(np.mean(np.any((sampled < lower) | (sampled > upper), axis=-1))),
                "clipping_mean_density_shift": (clipped-sampled).mean(axis=(0,1)).tolist(),
                "clipping_log_print_exposure_error": errors(fn(np.clip(subset, lower, upper)), fn(subset)),
                "grain_mean_density_shift": (sampled-patch).mean(axis=(0,1)).tolist(),
            })

    # Noise moments matter more than comparing individual random realizations.
    # Exact compound model: E[D]=p*Dmax; Var[D]=Dmax^2/N*p*(1-p*u*(1-1e-6)).
    for n in (1.0, 10.0, 100.0):
        for p in (0.01, 0.5, 0.99):
            density = np.full((512, 512), 2.2 * p, dtype=np.float32)
            saturation = 1 - p * 0.98 * (1 - 1e-6)
            expected_var = 2.2**2 / n * p * saturation
            row = {"particles_per_pixel": n, "development_probability": p,
                   "expected_mean": 2.2*p, "expected_variance": expected_var}
            for fast in (False, True):
                sample = layer_particle_model(density, 2.2, n, 0.98, seed=42, use_fast_stats=fast, use_gpu=False)
                row["fast" if fast else "scipy"] = {
                    "mean": float(sample.mean()), "variance": float(sample.var()),
                    "skewness": float(np.mean(((sample - sample.mean()) / sample.std())**3)),
                    "above_dmax_fraction": float(np.mean(sample > 2.2)),
                }
            report["grain_sampling"].append(row)

    impulse = np.zeros((513, 513), dtype=np.float32)
    impulse[256, 256] = 1
    for sigma in (1.0, 3.0, 10.0, 30.0):
        exact = gaussian_filter(impulse, sigma, truncate=3)
        fast = fast_gaussian_filter(impulse, sigma)
        report["gaussian"].append({"sigma_px": sigma, "l1_error": float(np.abs(fast-exact).sum()),
                                   "energy": float(fast.sum()), "center_ratio": float(fast[256,256]/exact[256,256])})
    q = np.geomspace(0.01, 30, 4096)  # dimensionless angular spatial frequency times lambda
    ideal = (1 + q*q)**(-1.5)
    for n, fit in _EXPONENTIAL_GAUSSIAN_FITS.items():
        mtf = np.sum(fit[:, 0, None] * np.exp(-0.5*(fit[:, 1, None]*q)**2), axis=0)
        report["exponential_mtf"][n] = errors(mtf, ideal)
    # Investigate a coefficient-only improvement, preserving DC gain and radial variance.
    fit_q = np.linspace(0, 8, 2048)
    def unpack(x):
        weights = np.exp(x[:3] - np.max(x[:3]))
        weights /= weights.sum()
        sigmas = np.exp(x[3:])
        sigmas *= np.sqrt(3 / np.sum(weights*sigmas**2))
        return weights, sigmas
    def residual(x):
        weights, sigmas = unpack(x)
        return np.sum(weights[:,None]*np.exp(-0.5*(sigmas[:,None]*fit_q)**2), axis=0) - (1+fit_q**2)**(-1.5)
    initial = _EXPONENTIAL_GAUSSIAN_FITS[3]
    fitted = least_squares(residual, np.log(np.concatenate([initial[:,0],initial[:,1]])))
    weights, sigmas = unpack(fitted.x)
    candidate = np.sum(weights[:,None]*np.exp(-0.5*(sigmas[:,None]*q)**2), axis=0)
    report["exponential_mtf"]["candidate_3_same_pass_count"] = {
        "fit": np.stack([weights, sigmas], axis=-1).tolist(),
        "error": errors(candidate, ideal),
        "weighted_sigma_squared": float(np.sum(weights*sigmas**2)),
        "note": "Analytic MTF fit only; discrete filtering and measured film validation still required.",
    }
    yy, xx = np.mgrid[-256:257, -256:257]
    discrete = []
    for decay in (5.0, 20.0):
        ideal_psf = np.exp(-np.hypot(xx, yy)/decay)
        ideal_psf /= ideal_psf.sum()
        row = {"decay_px": decay}
        for name, fit in (("current", initial), ("candidate", np.stack([weights,sigmas], -1))):
            psf = sum(float(a)*fast_gaussian_filter(impulse, float(s)*decay) for a,s in fit)
            row[name] = {"l1_error": float(np.abs(psf-ideal_psf).sum()),
                         "center_ratio": float(psf[256,256]/ideal_psf[256,256])}
        discrete.append(row)
    report["exponential_mtf"]["discrete_psf"] = discrete
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(args.out, flush=True)


if __name__ == "__main__":
    main()
