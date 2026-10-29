# Process BRAW video with spektrafilm

Use `scripts/process_braw.py` to decode a BRAW clip, apply a film/print
simulation, and export a MOV file. The script processes one frame at a time,
keeps the source frame rate and timecode, and preserves or precisely trims
audio. It currently runs on macOS.

The Python script drives a small companion C++ SDK adapter,
`scripts/braw_decode.cpp`. Keep both files together. The adapter is compiled
automatically on first use and cached; you do not need to build it manually.

## Requirements

Run commands from the root of this repository, using the project's Python
environment. This checkout already has the required environment and SDK.

For a fresh checkout:

```sh
uv sync
```

Also install:

- **Blackmagic RAW SDK for macOS**, including its `Mac` and `Samples` folders.
  The RAW Player alone is insufficient. The default location is
  `/Applications/Blackmagic RAW/Blackmagic RAW SDK`.
- **Apple command-line tools**, which provide `clang++`. If missing, run
  `xcode-select --install`.
- **FFmpeg and ffprobe** on `PATH`, with `libx264` and `prores_ks` encoders.
  A Homebrew installation can be made with `brew install ffmpeg`.

No additional Python dependencies are needed beyond spektrafilm's normal
runtime dependencies. The SDK's headers and libraries remain in their installed
location; they are not bundled into the Python project.

## Quick start with your sample

First render ten frames at a small size:

```sh
.venv/bin/python scripts/process_braw.py \
  "/Users/em/Downloads/The-End-of-the-World-Beagle-Channel-Original.braw" \
  -o build/video/beagle_test.mov \
  --scale 8 --frames 10 --verify-audio
```

Render the whole clip at the default quarter resolution:

```sh
.venv/bin/python scripts/process_braw.py \
  "/Users/em/Downloads/The-End-of-the-World-Beagle-Channel-Original.braw" \
  -o build/video/beagle_portra.mov \
  --verify-audio
```

Defaults are Portra 400, Portra Endura paper, fixed zero-stop exposure,
quarter-resolution decoding, and an H.264 display preview. On this sample the
quarter-resolution picture is 1536 x 864. Film grain is enabled; stochastic
glare remains disabled. Other spatial effects retain the profile defaults.

The script prints progress and produces:

```text
beagle_portra.mov             rendered video and audio
beagle_portra.mov.json        effective settings, timing and export checks
beagle_portra.mov.decode.log  SDK settings and frame/decode information
beagle_portra.mov.encode.log  FFmpeg errors, if any
```

Existing output files or sidecars are protected. Choose another output name,
or add `--overwrite` to allow replacing them. A new movie is rendered to a
temporary path and published only after successful export checks. Log files
are kept if an export fails. With `--overwrite`, logs can be updated even if
rendering fails; the existing movie remains until a new one passes checks.

## Full resolution and 10-bit ProRes

```sh
.venv/bin/python scripts/process_braw.py input.braw \
  -o build/video/film_prores.mov \
  --scale 1 --codec prores
```

`--codec prores` writes ProRes 422 HQ, 10-bit 4:2:2, with a 16-bit RGB pipe
feeding FFmpeg. `--codec preview` writes H.264, 8-bit 4:2:0; its default quality
is `--crf 18`, and lower CRF values increase quality and file size.

Both outputs are **display-referred sRGB**, with Rec.709 primaries and YUV
matrix tagging. RGB values outside [0, 1] are clipped during video encoding.
ProRes provides a higher-precision display export; it is not a scene-linear,
log, or RAW master. The input remains floating point through decoding, colour
conversion and simulation.

`--scale` divides both image dimensions: `1` is full resolution, `2` half,
`4` quarter, and `8` eighth. Full resolution processes sixteen times as many
pixels as the default quarter resolution and needs more memory and time.
The prior experiment, with grain disabled, measured about 0.73 seconds per
simulated frame at quarter resolution; full-resolution simulation has not been
benchmarked.

## Change the film, paper or exposure

```sh
.venv/bin/python scripts/process_braw.py input.braw \
  -o build/video/gold.mov \
  --film kodak_gold_200 \
  --paper kodak_portra_endura \
  --exposure-ev -0.5
```

Exposure is fixed for the job; per-frame auto-exposure is disabled. In film
mode, this controls the simulation's camera exposure compensation. In source
mode, it multiplies linear RGB by `2 ** exposure_ev` before display conversion.

### Grain

Film grain is enabled by default using the selected film profile's settings.
Each absolute source frame receives a different seed, so grain changes during
playback. With the same input, scale, settings and software environment, a base
seed makes the result reproducible: rendering a cut or retrying a frame uses
the same grain as rendering it within the whole clip.

Use `--grain-seed 123` for another grain sequence, or `--no-grain` to disable it.
Source mode bypasses the film simulation and adds no grain. Seeded grain uses
reference CPU samplers and a local random generator for both particles and
microstructure, independently of the parallel Numba/GPU random states. Grain
adds processing time; the earlier grain-disabled timings do not apply to it.

List profile IDs using the existing profile tools:

```sh
.venv/bin/spektrafilm-lut list film
.venv/bin/spektrafilm-lut list print
```

`--no-spatial-effects` disables the runtime's spatial effects, including
halation and diffusion. `--lut-resolution 17`, `33` or `65` selects the internal
enlarger/scanner LUT resolution; the default is 33. These internal LUTs
accelerate parts of the physical simulation.

## Render a section

```sh
.venv/bin/python scripts/process_braw.py input.braw \
  -o build/video/section.mov \
  --start-frame 25 --frames 50 --verify-audio
```

Frame indices are zero-based. This renders frames 25 through 74. Omit
`--frames` to render from the selected start through the end of the clip.
The SDK supplies the timecode of the selected starting frame, so timecode
also advances correctly when trimming.

For a whole clip, audio streams are copied without re-encoding. For a section,
audio is trimmed at the nearest sample boundaries and exported as PCM; the
usual BRAW 24-bit PCM format is retained. `--verify-audio` checks payload
hashes against the original audio or its exact sample trim. It adds an extra
read of the source/export. `--no-audio` exports video only.

Subclip audio with a nonzero stream start timestamp is currently rejected;
`--no-audio` can be used for such a clip. This avoids silently making a
misaligned audio cut.

## Inspect or convert without a film look

Show active SDK dimensions, frame count, colour generation, timecode and audio
metadata without rendering:

```sh
.venv/bin/python scripts/process_braw.py input.braw --info
```

Make a technical source preview, bypassing film simulation:

```sh
.venv/bin/python scripts/process_braw.py input.braw \
  -o build/video/source.mov --mode source
```

Source mode performs a colour-managed conversion to sRGB without a creative
look or highlight tone mapping. Bright display highlights can therefore clip.

## SDK location and colour processing

Specify a non-default SDK location with a quoted path:

```sh
.venv/bin/python scripts/process_braw.py input.braw \
  -o build/video/output.mov \
  --sdk "/path/to/Blackmagic RAW SDK"
```

The compiled adapter is cached under `build/braw_tools` in the checkout.
Changes to the adapter, SDK headers, dispatch code or target architecture
select a new cached binary. Use `--cache-dir /another/writable/path` to put
that cache elsewhere. `--threads 4` is the default for CPU decoding and video
encoding; `--threads 0` selects automatic thread counts. GPU decoding is not
implemented in this script.

By default, the colour path is:

```text
BRAW -> SDK Linear / Blackmagic Design / float32 RGB
     -> Blackmagic Wide Gamut to ACES2065-1, CAT02 adaptation
     -> spektrafilm -> encoded sRGB -> MOV
```

This external gamut conversion is restricted to SDK Generation 5 colour
science, which was tested with the supplied Pocket Cinema Camera 6K sample.
For SDK-managed ACES AP0 output instead, use:

```sh
.venv/bin/python scripts/process_braw.py input.braw \
  -o build/video/sdk_aces.mov --decode-space aces
```

That route lets the SDK handle the camera-to-ACES conversion for supported
cameras and generations. Other camera/generation combinations have not been
independently validated here. The tested SDK colour paths differ in a small
bright-highlight region; neither route has been checked against a calibrated
scene or an independent Resolve reference. See [the experiment findings](braw-processing.md).

RAW exposure is set to zero stops, embedded/sidecar creative LUT application,
highlight recovery and SDK gamut compression are disabled. White balance and
ISO come from the SDK's frame metadata. The SDK automatically loads a matching
sidecar when present; `--info` and the report indicate whether one is attached.
The script changes cloned processing attributes and does not write the BRAW
file or its sidecar.

## Errors and cancellation

Press **Ctrl-C** to cancel. Owned decoder/encoder processes are stopped and an
incomplete movie is not published. Exit codes are 0 for success, 1 for runtime
errors, 2 for invalid CLI arguments and 130 for cancellation.

For decode or encode failures, read the adjacent `.decode.log` and
`.encode.log`. Typical setup errors are a Player-only installation instead of
the SDK, an incorrect `--sdk` root, missing command-line tools, or missing
FFmpeg encoders. Use `--help` for the full option list.

## Verification performed

On the provided sample, the reusable script passed:

- SDK inspection and cached adapter compilation.
- A three-frame film cut starting at frame 10: correct adjusted timecode,
  exact PCM sample count and matching trimmed-audio hash.
- A two-frame ProRes source cut through SDK ACES output: 10-bit 4:2:2,
  correct colour tags/timecode and matching trimmed-audio hash.
- Whole-clip source conversion: 95 frames, with the original audio payload
  copied and verified.
- A silent film export with an alternate film, exposure adjustment and
  spatial effects disabled.
- Existing-output protection, including a failed replacement attempt that
  left the old movie unchanged.
- Actual Ctrl-C cancellation: exit code 130 and no published partial movie.
- Seeded grain: identical input/seed reproduces exactly, different frame seeds
  change the grain, and particle/microstructure sampling leaves NumPy's global
  random state untouched. Per-frame seed overrides are restored after errors.

The protocol/lifecycle tests can be run with:

```sh
.venv/bin/python -m pytest tests/test_process_braw.py -q
```
