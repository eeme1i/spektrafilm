# Sony RAW resolution investigation — 2026-10-11

The Sony file does show worse-than-pixel-count scaling on this machine. Large retained buffers and output-color temporary arrays are credible causes. GPU grain sampling is a small part of the measured scan. The reported GUI minute was not reproduced exactly: headless runs using the specified film/print pair took approximately 18–28 seconds at full resolution.

## Setup

Input: `/Users/em/Downloads/img/ILCE-7C.ARW`, decoded to 6,024 × 4,024 RGB (24.24 MP). A half-width/half-height version of the same decoded image supplies the 6.06 MP control, isolating resolution from camera/image-content differences. This is not a comparison with another camera's RAW file; no original lower-resolution reference RAW was supplied.

Film: Vision3 50D. Print: Kodak 2393, interpreted as the bundled stock corresponding to the supplied name “kodak_2939.” Ordinary runtime rendering defaults with GUI-equivalent fast settings: 17³ enlarger/scanner LUTs, fast grain sampling, Metal enabled, full-resolution mode, CAM16-UCS output gamut compression, ProPhoto RGB linear input, sRGB output. Custom GUI controls were not captured.

Hardware: MacBookAir10,1, arm64, 8 GB RAM, eight Numba threads. MLX 0.32.2 is installed. Metal is unavailable inside the tool sandbox, so the matching GPU measurements ran outside it. GPU sampling was available in these runs. RAW import and explicit kernel warmup were timed separately and excluded from scan totals.

Before the memory changes, RAW import normally took approximately 1.6–2.9 seconds. Color conversion returned **float64**, despite the importer's float32 documentation. The original input therefore occupied approximately 582 MB, versus 291 MB as float32. The simulation itself converts input to float32 during preprocessing. The importer now casts the converted result to float32.

Headless measurements exclude GUI setup/display work and GUI-resident output layers. Timings vary with memory pressure, background activity, and warmed caches; exploratory runs are not an exact reconstruction of the user's GUI session. An early concurrent CPU diagnostic was stopped and its results are excluded from the matching-stock comparison.

## Results

The warmed 6.06 MP control took 3.61 seconds. Full-resolution matching-stock runs took 18.39 and 23.34 seconds. An instrumented exploratory run took 28.49 seconds. A final sequential baseline/experiment comparison, without another diagnostic job running alongside either scan, gave:

| Stage | Previous runtime, 24.24 MP | Memory experiments, 24.24 MP |
| --- | ---: | ---: |
| Preprocess | 0.36 s | 0.36 s |
| Film exposure / scatter / halation | 5.71 s | 6.29 s |
| Development / inhibition / grain | 4.92 s | 4.12 s |
| Print exposure | 1.32 s | 1.28 s |
| Print development | 0.58 s | 0.44 s |
| Scan / output color / sharpening | 7.13 s | 5.12 s |
| **Total** | **20.03 s** | **17.61 s** |

The final combined experiment saved about 12%. Earlier repeated experiment scans took approximately 15.7–16.8 seconds. These are useful candidates, not a promise that a GUI minute will become 17 seconds. The modest and variable effect of releasing buffers alone is recorded separately.

## What changes with resolution

### Intermediate images accumulate

Previously, `runtime/topology.py:run_topology` kept every tap value in its state dictionary until the entire run finished. The observed full-resolution buffers were:

| Point in pipeline | Retained tap buffers, excluding temporary working arrays |
| --- | ---: |
| After preprocessing | 873 MB |
| After film exposure | 1,164 MB |
| After film development | 1,454 MB |
| After print exposure | 1,745 MB |
| Before scanning | 2,036 MB |
| After scanning | 2,327 MB |

All simulation outputs are float32; the imported input is float64. Most earlier taps have no remaining consumer by the time scanning starts. Retaining them unnecessarily raises the memory floor for a scanner that also needs large working arrays. At 6 MP these retained buffers are four times smaller. This is a plausible threshold effect on an 8 GB machine, though no per-process swap attribution was recorded.

### Gamut conversion makes large temporary arrays

The CAM16-UCS path constructs full-frame XYZ, perceptual-coordinate, polar-coordinate, lookup, and reconstruction arrays. Performing this per-pixel operation in bounded row blocks reduces temporary allocation without changing the selected color model. In the final pair, the complete scanner stage fell from 7.13 to 5.12 seconds; noise and memory conditions preclude attributing every saved second to one intervention.

### Spatial scales and grain branch choices change too

Pixel size is approximately 5.81 µm at full resolution and 11.62 µm at half resolution, for the default virtual 35 mm long edge. Physical blur radii expressed in pixels therefore double. Some Gaussian-mixture components cross the FIR/IIR dispatch threshold. The number of film records, grain sublayers, halation bounces, and LUT resolution remain the same.

Grain particles per pixel fall by four at full resolution, changing some Poisson/binomial algorithm branches. Nevertheless, measured GPU layer sampling plus its per-layer particle blur totaled only approximately 0.51–0.86 seconds for the full-resolution image, versus 0.11–0.12 seconds at half resolution. GPU grain sampling does not explain a minute by itself. The remainder of development includes inhibition diffusion, layer interpolation, microstructure, and grain blur.

The preceding density-conservation fix adds only small-table work; it adds no image-sized pass or per-pixel normalization. Its measured setup overhead was about 0.1 ms in the earlier investigation.

## Implemented memory changes

The three benchmark experiments are now implemented in the production runtime:

1. Keep the decoded input as float32, matching the precision the pipeline already uses.
2. Release a tap after its last planned consumer, retaining values required by later branches or the requested collection point.
3. Run output gamut conversion in blocks of at most 786,432 pixels (roughly 128 rows of this Sony image), leaving spatial glare and sharpening as full-image operations. A pixel limit also bounds work for unusually wide images and arbitrary batch shapes.

The combined prototype produced bit-identical deterministic output on a 252 × 377 patch from the supplied image, crossing a row-block boundary. Production tests cover buffer lifetimes, shared taps, repeated reads, aliased outputs, RAW conversion with linear/encoded output, and all five gamut algorithms in float32/float64 with contiguous, strided, and batch shapes. CAM16-UCS float32 has batch-size-dependent rounding differences of up to 2.4e-7 in the test corpus (bounded by a 5e-7 absolute tolerance); the other tested combinations are bit-identical. Image noise realizations were not compared pixel-for-pixel.

These changes preserve the existing film/color mathematics. Headless timing cannot attribute the actual GUI minute to Twitch or distinguish it from custom settings, acceleration fallback, cold compilation, or display preparation.

Production verification without experiment flags took **18.59 s** and **16.39 s** for the full-resolution Sony scan. The scanner stage took 5.18 s and 4.95 s. Import took 2.22 s and returned float32. Before the scanner runs, the dispatcher retains only the current 291 MB print-density image; immediately after it runs, input and output total 582 MB before the consumed input is released. These counts exclude the caller's original image, GUI layers, and temporary working arrays. The earlier baseline of 20.03 s is a historical comparison under variable background conditions, not a controlled speedup guarantee.

Focused regression checks: **199 passed**, with one existing LUT-mode test deselected because it references the removed `GrainParams.mult_usm_amount` field (also absent in HEAD). The initial run confirmed that failure; the final run excluded it. `git diff --check` passed. Existing snapshot discrepancies from the earlier quality investigation were not regenerated.

## Reproduce

Use a Metal-capable execution environment for GPU timings:

```
uv run python scripts/probe_raw_scaling.py /Users/em/Downloads/img/ILCE-7C.ARW \
  --film kodak_vision3_50d --print kodak_2393

uv run python scripts/probe_raw_scaling.py /Users/em/Downloads/img/ILCE-7C.ARW \
  --film kodak_vision3_50d --print kodak_2393 --scales 1 --repeats 2
```

Recorded runs are in [sony-raw-scaling-results.json](sony-raw-scaling-results.json). No image exports were created.

The optional experiment flags remain available for investigating alternate row-block sizes; tap release and float32 import are now enabled without those flags.
