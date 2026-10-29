# BRAW processing experiment

For the reusable command-line script and usage instructions, see
[Process BRAW video](process-braw.md). This page records the original experiment.

The sample successfully decoded through Blackmagic RAW SDK 6.0 and rendered
through the current spektrafilm runtime. All 95 frames were exported at
1536 x 864 and 25 fps, with the original PCM audio and timecode. The existing
film simulation accepted the decoded linear RGB arrays without runtime changes.

## Outputs

Generated files are in the git-ignored `build/braw_probe/` directory:

- `portra400_preview.mov`: Portra 400 / Portra Endura render of the entire clip.
- `source_preview.mov`: technical source-to-sRGB conversion for comparison.
- `comparison.png`: frame 47, source and film render side by side.
- `frame47_linear_aces.exr`: quarter-resolution float32 ACES2065-1 input.
- `experiment_results.json`: validation, settings, timings, and export checks.
- `full_resolution_colour_check.json`: additional 6K colour comparison.
- Decode logs and output metadata retain the effective SDK settings.

The MOVs are H.264, 8-bit 4:2:0 display previews with sRGB transfer and Rec.709
primaries/matrix. Out-of-range RGB values are clipped when making those display
copies. They are not high-precision finishing masters. The EXR preserves the
scene-linear input, including values above 1.0, and round-trips exactly.

## Source and effective settings

Source: `The-End-of-the-World-Beagle-Channel-Original.braw`, 159,349,723 bytes,
Pocket Cinema Camera 6K, Q5, Generation 5 colour science. The container reports
6176 x 3472; the SDK's active picture is 6144 x 3456. Duration is 3.8 seconds,
95 frames at 25 fps. Start timecode: `15:58:21:13`.

The decoder uses CPU processing with four threads and RGBF32 output. Gamma is
`Linear`; gamut is `Blackmagic Design`, corresponding to Blackmagic Wide Gamut
for this Gen 5 sample. Exposure is explicitly zero stops. Highlight recovery,
post-3D-LUT application, and SDK gamut compression are disabled. No sidecar is
attached. As-shot metadata stays constant across the clip: 5600 K, tint 10,
ISO 400. The helper changes cloned processing attributes; it does not save
metadata or modify the source clip.

The processing path is:

```text
BRAW SDK: Linear / Blackmagic Design / RGBF32
  -> Blackmagic Wide Gamut to ACES2065-1, CAT02 adaptation
  -> persistent spektrafilm Simulator
  -> sRGB preview encoding and FFmpeg export
```

Simulation uses Portra 400 / Portra Endura, fixed exposure, grain off, film and
print glare off, and 33-point internal enlarger/scanner LUTs. Deterministic
spatial effects retain profile defaults. This uses the current working tree's
model and profiles, not a frozen upstream reference.

## Numerical checks and colour limitation

- A +1-stop SDK decode is exactly twice the zero-stop linear decode.
- Inverting the SDK's Gen 5 Film output with colour-science agrees with its
  linear Blackmagic Wide Gamut decode: mean absolute error 1.77e-7, maximum
  9.64e-6 on the sampled frame.
- External conversion from Blackmagic Wide Gamut to ACES AP0 agrees closely
  with direct SDK AP0 output for most components: the 99.9th-percentile absolute
  difference is 9.54e-7. A small bright tail differs substantially, up to 0.469.
  SDK AP1 versus AP0 comparisons show a similar bright-tail discrepancy.
- The camera-wide-gamut linear output reaches 10.895. External ACES conversion
  reaches 11.364 at quarter resolution and 11.366 at full resolution.
- Applying a simple final clamp does not reproduce all the SDK's differences,
  including at full resolution. Clipping within SDK processing is a plausible
  explanation, but its exact location/order has not been established.
- All sequence renders are finite. Repeated rendering of the same input with
  these settings is identical.

The experiment uses camera-wide-gamut linear decode followed by an explicit
external colour conversion so its working-space transformation is inspectable.
Retaining larger transformed values is not proof of better scene fidelity.
There is no independent Resolve comparison or gray-chart calibration yet, and
this test does not establish absolute scene exposure or recover sensor clipping.

## Performance on this machine

Measured final 95-frame pass at 1536 x 864:

| Operation | Mean per frame | 95th percentile |
| --- | ---: | ---: |
| SDK read/decode/process and buffer copy, CPU | 16 ms | 27 ms |
| spektrafilm simulation | 734 ms | 996 ms |

The full pass took 86.8 seconds, including external colour conversion and two
preview encodes: about 1.09 source frames per second. Simulation alone ran at
about 1.36 frames per second. A separate single full-resolution SDK decode took
205 ms; full-resolution film simulation was not benchmarked.

These are diagnostic measurements for this clip and configuration. The main
bottleneck here is the simulation. Real-time playback would need a faster
simulation path, proxies, caching, or a combination of these.

## Export verification

Both final movies decode cleanly with FFmpeg. Each has 95 frames, 25 fps,
3.8-second duration, the source start timecode, and explicit colour tags.
SHA-256 hashes of the copied audio payload match the source exactly. The
source audio is stereo 48 kHz, 24-bit PCM.

## Reproduce

Requires the macOS Blackmagic RAW SDK, command-line clang++, FFmpeg/ffprobe,
and the project's Python environment. The SDK is used from its installed
location; its libraries and headers are not copied into the repository.

```sh
.venv/bin/python scripts/probe_braw.py \
  /Users/em/Downloads/The-End-of-the-World-Beagle-Channel-Original.braw
```

For a quick check, add `--frames 1 --out build/braw_probe/smoke`. Use
`--sdk /path/to/Blackmagic RAW SDK` for another installation and `--scale 8`
for a smaller preview. `--scale 1` requests full-resolution frames and will
require substantially more simulation memory and time. Output names are reused
on subsequent runs.

`scripts/braw_decode.cpp` is a small SDK callback adapter compiled by the
runner. Its stdout carries frame dimensions, frame index and float RGB pixels.
The runner consumes frames in order with bounded buffering, reuses a single
Simulator, and streams results into FFmpeg. The final runner also passed a
one-frame smoke run with its frame, timecode and colour-tag assertions.

## Integration direction

Keep decoding outside the physical simulation runtime. A future reader should
return a frame array plus explicit colour space, transfer function, timestamps,
and effective RAW settings. Keep the native SDK dependency optional. Before
adding a GUI video workflow, resolve the highlight conversion differences
against an independent reference, implement explicit per-frame grain seeds,
and add cancellation/error handling appropriate for long exports.
