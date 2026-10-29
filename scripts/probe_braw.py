"""Experimental macOS BRAW decode, colour checks and streaming film render.

Run with the project's Python environment; requires the installed Blackmagic
RAW SDK, clang++, and FFmpeg. Outputs stay under build/braw_probe by default.
This is an offline experiment, not a production video import API.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import plistlib
import re
import struct
import subprocess
import tempfile
from time import perf_counter

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "spektrafilm-braw-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "spektrafilm-braw-cache"))

import colour
import numpy as np
import OpenImageIO as oiio
from PIL import Image, ImageDraw

from spektrafilm import Simulator, digest_params, init_params


def read_exact(stream, size):
    data = bytearray()
    while len(data) < size:
        chunk = stream.read(size - len(data))
        if not chunk:
            if not data:
                return None
            raise RuntimeError(f"Truncated decoder output: {len(data)} / {size} bytes")
        data.extend(chunk)
    return data


def read_frame(stream):
    header = read_exact(stream, 16)
    if header is None:
        return None
    width, height, index = struct.unpack("<IIQ", header)
    if not (0 < width <= 32768 and 0 < height <= 32768):
        raise RuntimeError(f"Invalid decoder dimensions: {width} x {height}")
    data = read_exact(stream, width * height * 12)
    if data is None:
        raise RuntimeError("Decoder produced a header without pixels")
    return index, np.frombuffer(data, dtype="<f4").reshape(height, width, 3)


def metrics(image):
    return {
        "shape": list(image.shape), "finite": bool(np.isfinite(image).all()),
        "min": float(image.min()), "max": float(image.max()),
        "percentiles": dict(zip(("1", "50", "90", "99", "99.9"),
                                map(float, np.percentile(image, [1, 50, 90, 99, 99.9])))),
        "fraction_components_above_one": float(np.mean(image > 1)),
        "fraction_components_below_zero": float(np.mean(image < 0)),
    }


def difference(a, b):
    error = np.abs(a - b)
    return {"mean_abs": float(error.mean()), "p99_abs": float(np.percentile(error, 99)),
            "p99_9_abs": float(np.percentile(error, 99.9)), "max_abs": float(error.max())}


def camera_to_aces(rgb):
    # Gen 5 + SDK gamut "Blackmagic Design" is Blackmagic Wide Gamut.
    # CAT02 matches the SDK's D65-to-ACES-D60 conversion in this experiment.
    return colour.RGB_to_RGB(
        rgb, "Blackmagic Wide Gamut", "ACES2065-1",
        chromatic_adaptation_transform="CAT02",
    ).astype(np.float32)


def source_display(rgb):
    # Technical display conversion, without a creative look or highlight rolloff.
    return colour.RGB_to_RGB(rgb, "ACES2065-1", "sRGB", apply_cctf_encoding=True)


def uint8(rgb):
    if not np.isfinite(rgb).all():
        raise RuntimeError("Non-finite display pixels")
    return np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def save_exr(path, rgb):
    height, width, channels = rgb.shape
    spec = oiio.ImageSpec(width, height, channels, oiio.FLOAT)
    spec.attribute("oiio:ColorSpace", "ACES2065-1")
    spec.attribute("chromaticities", oiio.TypeDesc("float[8]"),
                   (0.7347, 0.2653, 0.0, 1.0, 0.0001, -0.077, 0.32168, 0.33767))
    output = oiio.ImageOutput.create(str(path))
    if not output or not output.open(str(path), spec):
        raise RuntimeError(f"Cannot create {path}")
    try:
        if not output.write_image(np.asarray(rgb, dtype=np.float32)):
            raise RuntimeError(output.geterror())
    finally:
        output.close()


def audio_hash(path):
    return subprocess.check_output([
        "ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0",
        "-c:a", "copy", "-f", "hash", "-hash", "sha256", "-",
    ], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip", type=Path)
    parser.add_argument("--sdk", type=Path, default=Path("/Applications/Blackmagic RAW/Blackmagic RAW SDK"))
    parser.add_argument("--out", type=Path, default=Path("build/braw_probe"))
    parser.add_argument("--scale", type=int, choices=(1, 2, 4, 8), default=4)
    parser.add_argument("--frames", type=int, default=None, help="Frames from clip start; default: whole clip")
    args = parser.parse_args()
    args.clip = args.clip.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(args.clip),
    ]))
    (args.out / "clip_metadata.json").write_text(json.dumps(metadata, indent=2))
    video = next(s for s in metadata["streams"] if s["codec_type"] == "video")
    total = int(video["nb_frames"])
    count = total if args.frames is None else args.frames
    if not 0 < count <= total:
        parser.error(f"--frames must be between 1 and {total}")
    middle = total // 2
    native = args.out / "decode"
    subprocess.run([
        "clang++", "-std=c++17", "-O2", "-I", str(args.sdk / "Mac/Include"),
        "-I", str(args.sdk / "Samples/Common"),
        str(Path(__file__).with_name("braw_decode.cpp")),
        str(args.sdk / "Mac/Include/BlackmagicRawAPIDispatch.cpp"),
        "-framework", "CoreFoundation", "-o", str(native),
    ], check=True)

    def command(start, length, gamma="Linear", gamut="Blackmagic Design", ev=0):
        return [str(native.resolve()), str(args.sdk / "Mac/Libraries"), str(args.clip),
                str(start), str(length), str(args.scale), gamma, gamut, str(ev)]

    def single(label, gamma="Linear", gamut="Blackmagic Design", ev=0):
        with (args.out / f"{label}.log").open("wb") as log:
            with subprocess.Popen(command(middle, 1, gamma, gamut, ev), stdout=subprocess.PIPE, stderr=log) as proc:
                frame = read_frame(proc.stdout)
                if proc.wait() or frame is None:
                    raise RuntimeError(f"Single-frame decode failed; see {label}.log")
                return frame[1]

    camera = single("validate_camera_linear")
    if "original_generation=5\n" not in (args.out / "validate_camera_linear.log").read_text():
        raise RuntimeError("This prototype's Blackmagic Wide Gamut mapping was validated for Gen 5 only")
    doubled = single("validate_camera_plus_one", ev=1)
    film_log = single("validate_film_gen5", "original", "original")
    ap0 = single("validate_ap0", gamut="ACES AP0")
    ap1 = single("validate_ap1", gamut="ACES AP1")
    aces = camera_to_aces(camera)
    film_decoded = colour.cctf_decoding(film_log, function="Blackmagic Film Generation 5")
    converted_ap1 = colour.RGB_to_RGB(ap1, "ACEScg", "ACES2065-1")
    report = {
        "clip": str(args.clip), "frame_count": total, "render_frame_count": count,
        "fps": video["avg_frame_rate"], "sample_frame": middle, "scale": args.scale,
        "decode": {"gamma": "Linear", "gamut": "Blackmagic Design", "generation": 5,
                   "format": "RGBF32", "exposure_ev": 0, "highlight_recovery": False,
                   "post_lut": "Disabled", "gamut_compression": False,
                   "white_balance": "as shot", "pipeline": "CPU", "threads": 4},
        "working_space": "ACES2065-1, converted from Blackmagic Wide Gamut with CAT02",
        "camera_linear": metrics(camera), "aces_linear": metrics(aces),
        "exposure_plus_one_vs_twice_zero": difference(doubled, 2 * camera),
        "film_gen5_inverse_vs_sdk_linear": difference(film_decoded, camera),
        "sdk_ap1_to_ap0_vs_sdk_ap0": difference(converted_ap1, ap0),
        "external_camera_to_ap0_vs_sdk_ap0": difference(aces, ap0),
        "external_camera_to_ap0_then_clamp_vs_sdk_ap0": difference(np.clip(aces, 0, float(camera.max())), ap0),
        "colour_limitation": "SDK output gamuts differ in a small bright pixel tail, consistent with clipping somewhere in SDK processing. A simple final clamp does not fully explain the differences. External conversion retains values above the SDK ACES output ceiling, but is not proof of better scene fidelity. Sensor clipping is not recoverable here.",
        "reference_limitation": "No independent Resolve decode or chart calibration; checks establish internal consistency only.",
    }
    with (args.sdk / "Mac/Libraries/BlackmagicRawAPI.framework/Resources/Info.plist").open("rb") as f:
        info = plistlib.load(f)
    report["sdk_bundle_version"] = info["CFBundleShortVersionString"]
    if not report["camera_linear"]["finite"] or report["exposure_plus_one_vs_twice_zero"]["max_abs"] > 1e-5:
        raise RuntimeError("Linear decode validation failed")
    if report["film_gen5_inverse_vs_sdk_linear"]["max_abs"] > 2e-5:
        raise RuntimeError("Gen 5 log/linear consistency check failed")
    save_exr(args.out / f"frame{middle}_linear_aces.exr", aces)
    Image.fromarray(uint8(source_display(aces))).save(args.out / "source_preview.png")

    params = init_params("kodak_portra_400", "kodak_portra_endura")
    params.io.input_color_space = "ACES2065-1"
    params.io.input_cctf_decoding = False
    params.camera.auto_exposure = False
    params.film_render.grain.active = False
    params.film_render.glare.active = False
    params.print_render.glare.active = False
    params.settings.use_enlarger_lut = True
    params.settings.use_scanner_lut = True
    params.settings.lut_resolution = 33
    simulator = Simulator(digest_params(params))
    report["simulation"] = {"film": "kodak_portra_400", "print": "kodak_portra_endura",
                            "exposure_ev": 0, "auto_exposure": False, "grain": False,
                            "film_print_glare": False, "internal_lut_resolution": 33,
                            "spatial_effects": "deterministic spatial effects retain profile defaults"}
    before = perf_counter()
    rendered = simulator.process(aces.copy())
    report["single_frame_simulation_seconds"] = perf_counter() - before
    report["repeat_render_error"] = difference(rendered, simulator.process(aces.copy()))
    report["rendered_frame"] = metrics(rendered)
    Image.fromarray(uint8(rendered)).save(args.out / "portra400_preview.png")
    width, height = aces.shape[1], aces.shape[0]
    comparison = Image.new("RGB", (width * 2, height + 38), "#151515")
    comparison.paste(Image.fromarray(uint8(source_display(aces))), (0, 38))
    comparison.paste(Image.fromarray(uint8(rendered)), (width, 38))
    draw = ImageDraw.Draw(comparison)
    draw.text((16, 12), "Source: technical sRGB conversion (display highlights clipped)", fill="white")
    draw.text((width + 16, 12), "Portra 400 / Endura | fixed exposure | grain off", fill="white")
    comparison.save(args.out / "comparison.png")
    (args.out / "validation_results.json").write_text(json.dumps(report, indent=2))

    # A single decoder process streams bounded frames. Neither frames nor a
    # whole clip are accumulated in memory. FFmpeg copies source PCM audio.
    fps = video["avg_frame_rate"]
    timecode = video.get("tags", {}).get("timecode", "00:00:00:00")
    duration = count / (float(fps.split('/')[0]) / float(fps.split('/')[1]))

    def encoder(path, log):
        return subprocess.Popen([
            "ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
            "-video_size", f"{width}x{height}", "-framerate", fps, "-i", "pipe:0",
            "-i", str(args.clip), "-map", "0:v:0", "-map", "1:a:0?",
            "-vf", "scale=in_range=full:out_range=tv:out_color_matrix=bt709,"
                   "setparams=range=limited:color_primaries=bt709:color_trc=iec61966-2-1:colorspace=bt709",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-color_primaries", "bt709", "-color_trc", "iec61966-2-1", "-colorspace", "bt709",
            "-c:a", "copy", "-t", str(duration), "-timecode", timecode,
            "-movflags", "+faststart", str(path),
        ], stdin=subprocess.PIPE, stderr=log)

    runs = []
    before = perf_counter()
    with (args.out / "sequence_decode.log").open("wb") as decode_log, \
         (args.out / "encode_source.log").open("wb") as source_log, \
         (args.out / "encode_portra.log").open("wb") as film_log_file:
        decoder = subprocess.Popen(command(0, count), stdout=subprocess.PIPE, stderr=decode_log)
        encoders = [encoder(args.out / "source_preview.mov", source_log),
                    encoder(args.out / "portra400_preview.mov", film_log_file)]
        try:
            for expected in range(count):
                frame = read_frame(decoder.stdout)
                if frame is None or frame[0] != expected:
                    raise RuntimeError(f"Missing or out-of-order frame {expected}")
                aces_frame = camera_to_aces(frame[1])
                begin = perf_counter()
                result = simulator.process(aces_frame)
                seconds = perf_counter() - begin
                if not np.isfinite(result).all():
                    raise RuntimeError(f"Non-finite render at frame {expected}")
                encoders[0].stdin.write(uint8(source_display(aces_frame)).tobytes())
                encoders[1].stdin.write(uint8(result).tobytes())
                runs.append(seconds)
                if expected % 10 == 0 or expected == count - 1:
                    print(f"Rendered {expected + 1}/{count}; simulation {seconds:.3f}s", flush=True)
            if decoder.wait() != 0:
                raise RuntimeError("Decoder failed; see sequence_decode.log")
            for proc in encoders:
                proc.stdin.close()
            for proc in encoders:
                if proc.wait() != 0:
                    raise RuntimeError("FFmpeg failed; see encode logs")
        finally:
            for proc in [decoder, *encoders]:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait()
    elapsed = perf_counter() - before
    decode_times = list(map(float, re.findall(r"decode_seconds=([0-9.eE+-]+)",
                                             (args.out / "sequence_decode.log").read_text())))
    report["sequence"] = {
        "rendered_frames": len(runs), "wall_seconds": elapsed,
        "simulation_mean_seconds": float(np.mean(runs)),
        "simulation_p95_seconds": float(np.percentile(runs, 95)),
        "decode_mean_seconds": float(np.mean(decode_times)),
        "decode_p95_seconds": float(np.percentile(decode_times, 95)),
        "simulation_frames_per_second": len(runs) / sum(runs),
        "end_to_end_frames_per_second": len(runs) / elapsed,
        "simulation_per_frame_seconds": runs,
    }
    text = (args.out / "sequence_decode.log").read_text()
    report["sequence"]["as_shot_metadata_values"] = {
        name: sorted(set(re.findall(r"attribute=" + key + r" value=([^\n]+)", text)))
        for name, key in (("white_balance_kelvin", "77626b76"),
                          ("white_balance_tint", "7762746e"), ("iso", "6669736f"))
    }
    report["output_validation"] = {}
    source_has_audio = any(s["codec_type"] == "audio" for s in metadata["streams"])
    source_audio_hash = audio_hash(args.clip) if source_has_audio and count == total else None
    for stem in ("source_preview", "portra400_preview"):
        inspection = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json",
            str(args.out / f"{stem}.mov"),
        ]))
        (args.out / f"{stem}_metadata.json").write_text(json.dumps(inspection, indent=2))
        stream = next(s for s in inspection["streams"] if s["codec_type"] == "video")
        if int(stream["nb_frames"]) != count or stream["avg_frame_rate"] != fps:
            raise RuntimeError(f"Export timing mismatch in {stem}")
        if stream.get("tags", {}).get("timecode") != timecode:
            raise RuntimeError(f"Export timecode mismatch in {stem}")
        if stream.get("color_primaries") != "bt709" or stream.get("color_transfer") != "iec61966-2-1":
            raise RuntimeError(f"Export colour tags missing in {stem}")
        audio_matches = audio_hash(args.out / f"{stem}.mov") == source_audio_hash if source_audio_hash else None
        if audio_matches is False:
            raise RuntimeError(f"Audio payload mismatch in {stem}")
        report["output_validation"][stem] = {
            "frames": int(stream["nb_frames"]), "fps": fps, "timecode": timecode,
            "audio_payload_sha256_matches_source": audio_matches,
        }
    (args.out / "experiment_results.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report["sequence"].items()
                      if k != "simulation_per_frame_seconds"}, indent=2))


if __name__ == "__main__":
    main()
