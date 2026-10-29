# Quality investigation — 2026-10-11

There are credible quality improvements without returning to slow full-frame spectral calculations. The numerical probes favor grain density conservation, LUT boundary handling, and stock-specific chemistry over increasing LUT resolution. Fuji's fourth color-sensitive layer is a valid research direction, but needs calibration rather than an arbitrary extra pass.

The initial investigation added a diagnostic script and measured results only. The density-conservation follow-up below is now implemented; the JSON retains the original measurements. Existing local Natura/profile/filter edits are untouched.

## What was measured

Run `.venv/bin/python scripts/probe_quality.py --out /tmp/spektrafilm-quality-probe.json` (or `uv run python ...`). Results from this checkout are in [quality-probe-results.json](quality-probe-results.json). The script records dependency versions when run.

The deterministic color target contains 2,916 linear-sRGB patches: a 9³ RGB cube at multipliers 0.03, 0.18, 1, and 4. Portra 400, PRO 400H, and X-tra 400 were printed onto Portra Endura with ordinary profile/filter defaults. LUT mode disables spatial effects, noise, autoexposure, and scanner corrections. Both enlarger and scanner LUTs are compared with direct spectral integration through the same pipeline. All reference outputs on this target fell within sRGB. This measures numerical agreement with the current model, not agreement with physical film.

Grain probes use CPU sampling, including 256² uniform-density patches with stock defaults at a virtual 35 mm width / 8,064 pixels. Parallel Numba grain is not reproducibly seeded; measured percentages and moments will fluctuate. Timings concern a tiny target, include cache/JIT effects, and cannot predict full-resolution export time. No paired digital/physical-film photographs were available for fidelity assessment.

## Priorities

### 1. Preserve density when splitting grain layers

The three fitted density sublayers do not exactly sum to the normalized parent density curve. The runtime split reproduces this mismatch. Maximum absolute errors are approximately 0.0050 optical density for Portra 400, 0.0049 for PRO 400H, and 0.0060 for X-tra 400. Similar discrepancies occur across the other filming profiles. Therefore enabling grain introduces a small deterministic density change before sampling noise.

Candidate: normalize nonnegative sublayer fractions at each parent-density lookup, then distribute the requested total density through those fractions. Handle zero-density endpoints explicitly. Check that reconstructed density is conserved and that the grain variance curve remains appropriate. This should add little arithmetic to the existing shared lookup. The measured discrepancy is small; its visible benefit still needs A/B assessment.

Relevant code: `model/density_curves.py:DensityLayers`, `model/grain.py:apply_grain_to_density_layers`. A correction should operate on the sampled split, or update the fitted profile representation consistently; changing only the noise amplitude does not repair the deterministic discrepancy.

Implemented follow-up: normalize the small sublayer tables at each parent-density knot, preserving fitted proportions there. Linear interpolation then conserves parent density between knots. Order lookup knots and sublayer values together to accommodate small digitization reversals near curve endpoints. Keep endpoint clamping outside the parent range. Sampling maxima come from the same corrected tables. No image-sized normalization pass or new film parameters are introduced.

Validation: every bundled filming profile (21, including the local Natura profile) preserves density in float32 and float64 at curve knots, intermediate densities, and clamped endpoints. Additional checks cover fitted proportions, zero density, missing proportions, and the complete layered-grain path with stochastic sampling replaced by identity sampling. Focused grain, filming, pipeline, emulsion, and warmup checks: 89 passed, 18 skipped; GPU execution was unavailable. Three print-output snapshots already fail with the original implementation, with identical before/after output, so snapshots were not regenerated. A 1.57 MP split benchmark measured roughly 22–23 ms before/after, with approximately 0.09 ms extra table setup; this is not a full-export benchmark or evidence of a speedup.

### 2. Avoid clipping grain through the spectral LUT domain

`utils/fast_interp_lut.py` clamps input coordinates to the LUT grid. The enlarger upper bound is the smooth density-curve maximum. The Poisson-binomial grain model has unbounded particle-count excursions, so noisy density can exceed that maximum even when its mean is within bounds.

At 95% of maximum density, approximately 0.4–0.6% of the tested pixels exceeded a bound in at least one channel. At maximum density, approximately 87% did. These are deliberately dense flat fields, not estimates of how often ordinary photos encounter the issue. On sampled pixels at maximum density, clipping alone changed print exposure by as much as approximately 0.09–0.12 log10 units. Subsequent print-curve saturation may reduce the final visible difference.

Candidate: give cached LUTs a fixed noise margin, or evaluate only out-of-domain pixels through direct integration. A sparse fallback preserves exact behavior for arbitrary excursions; a padded domain trades interpolation density for fewer fallbacks. Choose margins by inspecting real images and grain settings, rather than rebuilding the LUT from every image's extrema. Merely increasing 17³ to 33³ does not solve boundary clipping.

### 3. Calibrate chemistry per stock, especially Fuji

`runtime/params_builder.py:_apply_film_specifics` assigns the same inhibitor coefficients to all negative stocks. Profiles distinguish spectral sensitivities, dye spectra, and characteristic curves, but the runtime does not distinguish their negative-film inhibition matrices. This is a plausible limitation for stock identity and nonlinear color separation; it is not a measured fidelity failure.

Start with paired digital and physical-film chart exposures under daylight, tungsten, and fluorescent/LED illumination, plus an exposure bracket. Fit stock-specific inhibitor terms while constraining neutral-ramp behavior. Include skin, foliage, saturated blue-green colors, and mixed lighting. Fit on one subset and validate on different lighting and exposure combinations. Printing/scanning must be held fixed or characterized separately so their color effects are not absorbed into film chemistry.

### 4. Refine the blur approximation without more passes

The exponential blur uses a placeholder three-Gaussian mixture. Against the analytic normalized 2D exponential MTF, its maximum absolute transfer-function error is 0.01585. A coefficient-only fit preserving unit DC gain and the exponential's second spatial moment reduces that to 0.00449, using the same three Gaussian calls.

Candidate rows `(weight, sigma/decay)`:

```
0.0861027851  0.4629573071
0.5144845716  1.1913089365
0.3994126433  2.3741794185
```

Discrete impulse tests using the actual fast filter reduced kernel L1 error from 0.0802 to 0.0597 at a 5 px decay, and from 0.0673 to 0.0411 at 20 px. However, the central peak approximation became worse: at 20 px, the peak/ideal ratio changed from 0.877 to 0.840. This is a candidate for a joint PSF/MTF fit, not a validated replacement. Better approximation of an assumed exponential also does not prove better agreement with measured film.

The IIR Gaussian itself redistributes about 9.9% of impulse-kernel mass relative to the truncated direct Gaussian at sigma=3 px (L1 difference), falling to 6.7% at 10 px. Energy remains approximately conserved. Absolute per-pixel errors can be small while the shape difference is material. Consider measuring continuity across the FIR/IIR threshold before spending time on more exact large-radius blurs.

## Fuji's fourth layer

[Fujifilm's PRO 400H datasheet](https://asset.fujifilm.com/master/emea/files/2020-10/a6cb96275e4957ddc7b3ca932b7755e5/films_pro-400h_datasheet_01.pdf) explicitly describes a fourth color-sensitive layer, including improved color under fluorescent/mixed lighting. [Fujifilm's technology explanation](https://www.fujifilm.com.hk/products/professional_films/technology/) describes green-color correction that reduces red response when capturing green colors. [Older Superia X-tra 400 documentation](https://www.fujifilm.com.hk/products/consumer_film/pdf/superia_xtra400_datasheet.pdf) also advertises four-layer technology. Establish the emulsion generation before assigning this behavior to a Superia profile.

This is separate from the three speed/grain sublayers within each color record, and separate from ordinary cyan dye formed by the red-sensitive record. Adding a fourth grain sublayer or halation bounce would not model the correction.

Today the schema validates three sensitivity and dye channels; spectral upsampling/adaptation and the 3×3 inhibitor matrix also assume three responses. A research prototype could add an optional fourth spectral response and an exposure-dependent correction to the existing three dye-forming records. Treat that as a hypothesis to fit, not an established chemical model. Do not assume a fourth output dye is required. Published effective sensitivity/characteristic curves may already incorporate some correction, so avoid counting the effect twice.

The correction could be baked into a transform from input RGB to the three final records when spatial effects are absent. A spatial fourth donor would need an additional field and diffusion work. Either way, four sensitivity records do not imply four complete image-processing passes.

The RGB input cannot uniquely determine the original spectrum. Reconstructing a fourth response from the same RGB therefore adds a modeling assumption, not recovered measurement. This matters most for narrowband artificial light, precisely where the correction is interesting. A convincing result needs real-film calibration and evaluation on held-out lighting.

## Lower priority: bigger LUTs or exact grain sampling

Maximum end-to-end CIEDE2000 difference from direct spectral evaluation:

| Stock | 17³ | 33³ | 65³ |
| --- | ---: | ---: | ---: |
| Portra 400 | 0.0267 | 0.0035 | 0.0005 |
| PRO 400H | 0.0274 | 0.0031 | 0.0005 |
| X-tra 400 | 0.0267 | 0.0040 | 0.0005 |

The 17³ result is already a very small color difference on this test. Larger LUTs may be useful for unusual settings, but they are unlikely to be the main quality opportunity. Their memory/build cost rises cubically with resolution.

Fast CPU grain broadly preserved means and variances on the sampled parameter grid. The normal approximations reduce distribution skew near saturation: at one particle per pixel and development probability 0.99, fast skew was near zero versus about 0.17 for SciPy. This warrants a distribution/texture check if highly magnified saturated grain is a priority, but does not justify reverting the accelerator based on these probes. GPU behavior was not measured in this investigation.

Recommended sequence: density conservation and LUT boundary handling first; then calibrated stock-specific chemistry and a joint blur fit; finally a PRO 400H fourth-response prototype validated against physical film.
