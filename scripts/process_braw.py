#!/usr/bin/env python3
"""Decode BRAW and render it through spektrafilm (macOS).

Examples:
  python scripts/process_braw.py input.braw -o preview.mov
  python scripts/process_braw.py input.braw -o final.mov --scale 1 --codec prores
  python scripts/process_braw.py input.braw --info

The separately installed Blackmagic RAW SDK is required. The companion
braw_decode.cpp adapter is compiled automatically on first use.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import plistlib
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from time import perf_counter

DEFAULT_SDK = Path('/Applications/Blackmagic RAW/Blackmagic RAW SDK')
SCRIPT_DIR = Path(__file__).resolve().parent


class ProcessingError(RuntimeError):
    """An actionable input, dependency, decode, or export error."""


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('input', type=Path, help='Source .braw file')
    p.add_argument('-o', '--output', type=Path, help='Output .mov; required unless --info is used')
    p.add_argument('--info', action='store_true', help='Inspect SDK clip metadata without rendering')
    p.add_argument('--sdk', type=Path, default=DEFAULT_SDK, help='Blackmagic RAW SDK root directory')
    p.add_argument('--cache-dir', type=Path, default=SCRIPT_DIR.parent / 'build/braw_tools',
                   help='Cache for the compiled SDK adapter')
    p.add_argument('--scale', type=int, choices=(1, 2, 4, 8), default=4,
                   help='Decode divisor in each dimension: 1=full, 4=quarter (default)')
    p.add_argument('--start-frame', type=int, default=0, help='First frame, zero-based (default: 0)')
    p.add_argument('--frames', type=int, help='Number of frames; default: remaining clip')
    p.add_argument('--mode', choices=('film', 'source'), default='film',
                   help='Film simulation or technical source-to-sRGB conversion')
    p.add_argument('--film', default='kodak_portra_400', help='Film profile ID')
    p.add_argument('--paper', default='kodak_portra_endura', help='Print-paper profile ID')
    p.add_argument('--exposure-ev', type=float, default=0, help='Fixed exposure adjustment in stops')
    p.add_argument('--no-grain', action='store_true', help='Disable film grain (enabled by default)')
    p.add_argument('--grain-seed', type=int, default=0,
                   help='Base seed for reproducible, independent frame grain (default: 0)')
    p.add_argument('--no-spatial-effects', action='store_true', help='Disable spatial simulation effects')
    p.add_argument('--lut-resolution', type=int, choices=(17, 33, 65), default=33,
                   help='Internal enlarger/scanner LUT size (default: 33)')
    p.add_argument('--decode-space', choices=('blackmagic-gen5', 'aces'), default='blackmagic-gen5',
                   help='Linear BMD gamut + external ACES conversion, or SDK ACES AP0 output')
    p.add_argument('--codec', choices=('preview', 'prores'), default='preview',
                   help='H.264 8-bit preview or ProRes 422 HQ 10-bit (default: preview)')
    p.add_argument('--crf', type=int, default=18, help='H.264 quality, 0..51; lower is better (default: 18)')
    p.add_argument('--threads', type=int, default=4, help='SDK/encoder CPU threads; 0=automatic (default: 4)')
    p.add_argument('--no-audio', action='store_true', help='Export video only')
    p.add_argument('--verify-audio', action='store_true', help='Verify exported audio payload hashes')
    p.add_argument('--overwrite', action='store_true', help='Allow replacement of the output and its report/logs')
    return p


def run_json(command: list[str]) -> dict:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise ProcessingError(result.stderr.strip() or f'{command[0]} failed')
    try:
        return json.loads(result.stdout)
    except ValueError as exc:
        raise ProcessingError(f'{command[0]} returned invalid JSON') from exc


def probe(path: Path) -> dict:
    return run_json(['ffprobe', '-v', 'error', '-show_format', '-show_streams', '-of', 'json', str(path)])


def build_decoder(sdk: Path, cache: Path) -> Path:
    if sys.platform != 'darwin':
        raise ProcessingError('This script currently supports macOS only.')
    files = [SCRIPT_DIR / 'braw_decode.cpp', sdk / 'Mac/Include/BlackmagicRawAPI.h',
             sdk / 'Mac/Include/BlackmagicRawAPIDispatch.cpp', sdk / 'Samples/Common/PlatformHelpers.h']
    framework = sdk / 'Mac/Libraries/BlackmagicRawAPI.framework'
    for path in [*files, framework]:
        if not path.exists():
            raise ProcessingError(f'Missing SDK/helper file: {path}\nUse --sdk to specify the SDK root.')
    fingerprint = hashlib.sha256((str(sdk.resolve()) + platform.machine() + 'c++17-O2').encode())
    for path in files:
        fingerprint.update(path.read_bytes())
    cache.mkdir(parents=True, exist_ok=True)
    binary = cache / f'braw_decode_{fingerprint.hexdigest()[:20]}'
    if binary.is_file():
        return binary.resolve()
    compiler = shutil.which('clang++')
    if not compiler:
        raise ProcessingError('clang++ is missing. Install Apple command-line tools with: xcode-select --install')
    print('Compiling the Blackmagic RAW adapter...', file=sys.stderr)
    with tempfile.TemporaryDirectory(prefix='compile-', dir=cache) as directory:
        temporary = Path(directory) / 'decode'
        result = subprocess.run([
            compiler, '-std=c++17', '-O2', '-I', str(sdk / 'Mac/Include'),
            '-I', str(sdk / 'Samples/Common'), str(files[0]), str(files[2]),
            '-framework', 'CoreFoundation', '-o', str(temporary),
        ], capture_output=True, text=True)
        if result.returncode:
            raise ProcessingError(f'Cannot compile SDK adapter:\n{result.stderr.strip()}')
        temporary.replace(binary)
    return binary.resolve()


def decoder_command(args, binary: Path, start: int, count: int) -> list[str]:
    gamut = 'Blackmagic Design' if args.decode_space == 'blackmagic-gen5' else 'ACES AP0'
    return [str(binary), str(args.sdk / 'Mac/Libraries'), str(args.input),
            str(start), str(count), str(args.scale), 'Linear', gamut, '0', str(args.threads)]


def inspect(args, binary: Path) -> tuple[dict, str]:
    result = subprocess.run(decoder_command(args, binary, args.start_frame, 0), capture_output=True)
    log = result.stderr.decode(errors='replace')
    if result.returncode:
        raise ProcessingError(f'SDK could not open the clip:\n{log[-2500:]}')
    match = re.search(r'clip=(\d+)x(\d+) frames=(\d+) fps=([0-9.eE+-]+) sidecar=(\d+)', log)
    generation = re.search(r'^original_generation=(\d+)$', log, re.M)
    timecode = re.search(r'^start_timecode=(.+)$', log, re.M)
    if not match or not generation:
        raise ProcessingError('SDK adapter returned incomplete clip information.')
    width, height, frames, fps, sidecar = match.groups()
    return {'width': int(width), 'height': int(height), 'frames': int(frames),
            'sdk_fps': float(fps), 'generation': int(generation.group(1)),
            'sidecar_attached': bool(int(sidecar)),
            'start_timecode': timecode.group(1).strip() if timecode else None}, log


def read_exact(stream, size: int) -> bytearray | None:
    data = bytearray()
    while len(data) < size:
        chunk = stream.read(size - len(data))
        if not chunk:
            if data:
                raise ProcessingError(f'Truncated decoder data ({len(data)} of {size} bytes).')
            return None
        data.extend(chunk)
    return data


def frame_grain_seed(base_seed: int, frame_index: int) -> int:
    """Stable grain for an absolute source frame, independent of render order/cuts."""
    payload = struct.pack('<QQ', base_seed, frame_index)
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8, person=b'braw-grain-v1').digest(), 'little')


def read_frame(stream):
    header = read_exact(stream, 16)
    if header is None:
        return None
    width, height, index = struct.unpack('<IIQ', header)
    if not (0 < width <= 16384 and 0 < height <= 16384):
        raise ProcessingError(f'Invalid decoded dimensions: {width} x {height}')
    pixels = read_exact(stream, width * height * 12)
    if pixels is None:
        raise ProcessingError('Decoder returned a frame header without pixels.')
    return index, width, height, pixels


@contextmanager
def process(command, **kwargs):
    child = subprocess.Popen(command, **kwargs)
    try:
        yield child
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        for stream in (child.stdin, child.stdout):
            if stream:
                try:
                    stream.close()
                except BrokenPipeError:
                    pass


def audio_options(args, metadata: dict, fps: Fraction, count: int, total: int):
    streams = [s for s in metadata['streams'] if s['codec_type'] == 'audio']
    if args.no_audio or not streams:
        return ['-an'], [], []
    full = args.start_frame == 0 and count == total
    options, descriptions, checks = [], [], []
    for index, stream in enumerate(streams):
        options += ['-map', f'1:a:{index}']
        if full:
            options += [f'-c:a:{index}', 'copy']
            descriptions.append({'stream': index, 'mode': 'copy', 'codec': stream['codec_name']})
            checks.append((index, None, 'copy'))
        else:
            # For a cut, trim PCM samples explicitly rather than cutting copied
            # packets at a potentially different boundary than the video.
            rate = int(stream['sample_rate'])
            if abs(float(stream.get('start_time', 0))) > 1 / rate:
                raise ProcessingError('Subclip audio with a nonzero start timestamp is not supported; use --no-audio.')
            start = round(Fraction(args.start_frame * rate) / fps)
            end = round(Fraction((args.start_frame + count) * rate) / fps)
            filter_ = f'atrim=start_sample={start}:end_sample={end},asetpts=PTS-STARTPTS'
            codec = stream['codec_name'] if stream['codec_name'].startswith('pcm_') else 'pcm_s24le'
            options += [f'-filter:a:{index}', filter_, f'-c:a:{index}', codec]
            descriptions.append({'stream': index, 'mode': 'sample-trimmed PCM', 'codec': codec,
                                 'start_sample': start, 'end_sample': end})
            checks.append((index, filter_, codec))
    return options, descriptions, checks


def audio_hash(path: Path, index: int, filter_: str | None = None, codec: str = 'copy') -> str:
    command = ['ffmpeg', '-v', 'error', '-i', str(path), '-map', f'0:a:{index}']
    if filter_:
        command += ['-af', filter_]
    command += ['-c:a', codec, '-f', 'hash', '-hash', 'sha256', '-']
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise ProcessingError(f'Audio verification failed: {result.stderr.strip()}')
    return result.stdout.strip()


def encoder_command(args, temporary: Path, width: int, height: int, fps: Fraction,
                    timecode: str, audio: list[str]) -> list[str]:
    is_prores = args.codec == 'prores'
    command = ['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pixel_format',
               'rgb48le' if is_prores else 'rgb24', '-video_size', f'{width}x{height}',
               '-framerate', str(fps), '-i', 'pipe:0', '-i', str(args.input), '-map', '0:v:0', *audio,
               '-vf', 'scale=in_range=full:out_range=tv:out_color_matrix=bt709,'
               'setparams=range=limited:color_primaries=bt709:color_trc=iec61966-2-1:colorspace=bt709']
    if is_prores:
        command += ['-c:v', 'prores_ks', '-profile:v', '3', '-pix_fmt', 'yuv422p10le']
    else:
        command += ['-c:v', 'libx264', '-preset', 'fast', '-crf', str(args.crf), '-pix_fmt', 'yuv420p']
    command += ['-threads', str(args.threads), '-color_primaries', 'bt709',
                '-color_trc', 'iec61966-2-1', '-colorspace', 'bt709',
                '-timecode', timecode, '-movflags', '+faststart', str(temporary)]
    return command


def render(args, binary: Path, info: dict, metadata: dict, fps: Fraction, count: int,
           inspection_log: str, temporary: Path, decode_log: Path, encode_log: Path) -> dict:
    # Keep --help and --info usable without importing the simulation stack.
    try:
        import colour
        import numpy as np
        from spektrafilm import Simulator, digest_params, init_params
    except ImportError as exc:
        raise ProcessingError(f'Python dependency unavailable: {exc}. Run with the spektrafilm environment.') from exc
    simulator = None
    if args.mode == 'film':
        try:
            params = init_params(args.film, args.paper)
        except (ValueError, KeyError, FileNotFoundError) as exc:
            raise ProcessingError(f'Cannot load film/paper profiles: {exc}') from exc
        params.io.input_color_space = 'ACES2065-1'
        params.io.input_cctf_decoding = False
        params.camera.auto_exposure = False
        params.camera.exposure_compensation_ev = args.exposure_ev
        params.debug.deactivate_stochastic_effects = False
        params.film_render.grain.active = not args.no_grain
        params.print_render.glare.active = False
        params.debug.deactivate_spatial_effects = args.no_spatial_effects
        params.film_render.glare.active = False
        params.settings.use_enlarger_lut = True
        params.settings.use_scanner_lut = True
        params.settings.lut_resolution = args.lut_resolution
        simulator = Simulator(digest_params(params))
    audio, audio_description, audio_checks = audio_options(args, metadata, fps, count, info['frames'])
    elapsed_simulation = 0.0
    started = perf_counter()
    print(f"Rendering {count} frames at {fps} fps; scale 1/{args.scale}, {args.mode}, {args.codec}", flush=True)
    with ExitStack() as stack:
        decoder_log = stack.enter_context(decode_log.open('w'))
        decoder_log.write(inspection_log)
        decoder_log.flush()
        encoder_log = stack.enter_context(encode_log.open('w'))
        decoder = stack.enter_context(process(decoder_command(args, binary, args.start_frame, count),
                                              stdout=subprocess.PIPE, stderr=decoder_log))
        encoder = None
        dimensions = None
        for offset in range(count):
            frame = read_frame(decoder.stdout)
            expected = args.start_frame + offset
            if frame is None or frame[0] != expected:
                raise ProcessingError(f'Decoder stopped or returned an unexpected frame at {expected}; see {decode_log}')
            _, width, height, buffer = frame
            if encoder is None:
                if width % 2 or height % 2:
                    raise ProcessingError('This video exporter requires even frame dimensions; try another --scale.')
                dimensions = (width, height)
                encoder = stack.enter_context(process(
                    encoder_command(args, temporary, width, height, fps, info['start_timecode'], audio),
                    stdin=subprocess.PIPE, stderr=encoder_log))
            if dimensions != (width, height):
                raise ProcessingError('Decoded frame dimensions changed within the clip.')
            rgb = np.frombuffer(buffer, dtype='<f4').reshape(height, width, 3)
            if not np.isfinite(rgb).all():
                raise ProcessingError(f'Non-finite decoded pixels in frame {expected}')
            if args.decode_space == 'blackmagic-gen5':
                rgb = colour.RGB_to_RGB(rgb, 'Blackmagic Wide Gamut', 'ACES2065-1',
                                        chromatic_adaptation_transform='CAT02').astype(np.float32)
            begin = perf_counter()
            if simulator:
                seed = frame_grain_seed(args.grain_seed, expected) if not args.no_grain else None
                rgb = simulator.process(rgb, grain_seed=seed)
            else:
                rgb = colour.RGB_to_RGB(rgb * (2 ** args.exposure_ev), 'ACES2065-1', 'sRGB',
                                        apply_cctf_encoding=True)
            elapsed_simulation += perf_counter() - begin
            if not np.isfinite(rgb).all():
                raise ProcessingError(f'Non-finite output pixels in frame {expected}')
            maximum, dtype = (65535, '<u2') if args.codec == 'prores' else (255, 'u1')
            encoded = np.rint(np.clip(rgb, 0, 1) * maximum).astype(dtype)
            try:
                encoder.stdin.write(encoded.tobytes())
            except BrokenPipeError as exc:
                raise ProcessingError(f'FFmpeg stopped accepting frames; see {encode_log}') from exc
            if offset == 0 or (offset + 1) % 10 == 0 or offset + 1 == count:
                print(f'  {offset + 1}/{count} frames ({perf_counter() - started:.1f}s)', flush=True)
        if decoder.wait(timeout=30):
            raise ProcessingError(f'SDK decoding failed; see {decode_log}')
        try:
            encoder.stdin.close()
        except BrokenPipeError as exc:
            raise ProcessingError(f'FFmpeg failed; see {encode_log}') from exc
        if encoder.wait(timeout=300):
            raise ProcessingError(f'FFmpeg encoding failed; see {encode_log}')
    exported = probe(temporary)
    video = next(s for s in exported['streams'] if s['codec_type'] == 'video')
    if int(video.get('nb_frames', 0)) != count or Fraction(video['avg_frame_rate']) != fps:
        raise ProcessingError('Export frame count or frame rate does not match the request.')
    if (video['width'], video['height']) != dimensions:
        raise ProcessingError('Export dimensions do not match the decoded frames.')
    if video.get('tags', {}).get('timecode') != info['start_timecode']:
        raise ProcessingError('Export timecode does not match the source start frame.')
    if video.get('color_transfer') != 'iec61966-2-1' or video.get('color_primaries') != 'bt709':
        raise ProcessingError('Export colour tags are missing or incorrect.')
    output_audio = [s for s in exported['streams'] if s['codec_type'] == 'audio']
    if len(output_audio) != len(audio_description):
        raise ProcessingError('Export audio stream count is incorrect.')
    for description, stream in zip(audio_description, output_audio):
        if description['codec'] != stream['codec_name']:
            raise ProcessingError('Export audio codec is incorrect.')
        if description['mode'] == 'sample-trimmed PCM':
            wanted = description['end_sample'] - description['start_sample']
            duration_samples = Fraction(stream['duration_ts']) * Fraction(stream['time_base']) * int(stream['sample_rate'])
            if duration_samples != wanted:
                raise ProcessingError(f'Export audio sample count mismatch: {duration_samples} vs {wanted}')
    if args.verify_audio:
        for description, (index, filter_, codec) in zip(audio_description, audio_checks):
            expected = audio_hash(args.input, index, filter_, codec)
            actual = audio_hash(temporary, index)
            if expected != actual:
                raise ProcessingError(f'Audio payload verification failed for stream {index}')
            description['sha256_matches_source_or_exact_trim'] = True
    return {'frames': count, 'width': dimensions[0], 'height': dimensions[1], 'fps': str(fps),
            'timecode': info['start_timecode'], 'elapsed_seconds': perf_counter() - started,
            'processing_seconds': elapsed_simulation, 'audio': audio_description,
            'video_codec': video['codec_name'], 'pixel_format': video['pix_fmt'],
            'colour_primaries': video['color_primaries'], 'transfer': video['color_transfer']}


def main(argv=None) -> int:
    p = parser()
    args = p.parse_args(argv)
    if not args.info and args.output is None:
        p.error('--output is required unless --info is used')
    if args.start_frame < 0 or (args.frames is not None and args.frames <= 0):
        p.error('--start-frame must be nonnegative and --frames must be positive')
    if args.threads < 0 or not 0 <= args.crf <= 51 or not math.isfinite(args.exposure_ev):
        p.error('--threads must be nonnegative, --crf must be 0..51, and exposure must be finite')
    if abs(args.exposure_ev) > 30:
        p.error('--exposure-ev must be between -30 and 30')
    if not 0 <= args.grain_seed < 2**64:
        p.error('--grain-seed must be between 0 and 18446744073709551615')
    args.input = args.input.expanduser().resolve()
    args.sdk = args.sdk.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    try:
        if not args.input.is_file():
            raise ProcessingError(f'Input file does not exist: {args.input}')
        for tool in ('ffmpeg', 'ffprobe'):
            if not shutil.which(tool):
                raise ProcessingError(f'{tool} is missing. Install FFmpeg and put it on PATH.')
        paths = []
        if not args.info:
            args.output = args.output.expanduser().resolve()
            if args.output.suffix.lower() != '.mov':
                raise ProcessingError('Use a .mov output path (supports both codecs and uncompressed source audio).')
            if args.output == args.input:
                raise ProcessingError('Output must differ from the input.')
            paths = [args.output, *[Path(str(args.output) + suffix) for suffix in ('.json', '.decode.log', '.encode.log')]]
            if not args.overwrite:
                for path in paths:
                    if path.exists():
                        raise ProcessingError(f'Output already exists: {path}. Choose another path or use --overwrite.')
        binary = build_decoder(args.sdk, args.cache_dir)
        info, inspection_log = inspect(args, binary)
        metadata = probe(args.input)
        video = next(s for s in metadata['streams'] if s['codec_type'] == 'video')
        try:
            fps = Fraction(video['avg_frame_rate'])
            if fps <= 0:
                raise ValueError
        except (ValueError, ZeroDivisionError, KeyError):
            raise ProcessingError('Cannot determine a positive rational frame rate from the source container.')
        if abs(float(fps) - info['sdk_fps']) > 1e-3:
            raise ProcessingError('SDK and container frame rates disagree.')
        if args.info:
            print(json.dumps({**info, 'fps': str(fps), 'input': str(args.input),
                              'audio_streams': [s for s in metadata['streams'] if s['codec_type'] == 'audio']}, indent=2))
            return 0
        if args.decode_space == 'blackmagic-gen5' and info['generation'] != 5:
            raise ProcessingError('The external Blackmagic gamut conversion supports Gen 5 only. '
                                  'Use --decode-space aces for SDK-managed ACES output.')
        count = info['frames'] - args.start_frame if args.frames is None else args.frames
        if count <= 0 or args.start_frame + count > info['frames']:
            raise ProcessingError(f'Requested frame range exceeds the {info["frames"]}-frame clip.')
        if not info['start_timecode']:
            raise ProcessingError('SDK did not return a source timecode.')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output, report_path, decode_log, encode_log = paths
        # The final movie is replaced only after successful rendering and checks.
        with tempfile.TemporaryDirectory(prefix=f'.{output.stem}-', dir=output.parent) as directory:
            temporary = Path(directory) / 'render.mov'
            result = render(args, binary, info, metadata, fps, count, inspection_log,
                            temporary, decode_log, encode_log)
            with (args.sdk / 'Mac/Libraries/BlackmagicRawAPI.framework/Resources/Info.plist').open('rb') as f:
                sdk_version = plistlib.load(f).get('CFBundleShortVersionString', 'unknown')
            report = {'input': str(args.input), 'output': str(output), 'sdk_root': str(args.sdk),
                      'sdk_version': sdk_version, 'working_space': 'ACES2065-1',
                      'source': info, 'settings': {k: v for k, v in vars(args).items()
                       if k not in ('input', 'output', 'sdk', 'cache_dir', 'info')}, 'result': result,
                      'decode': {'gamma': 'Linear', 'exposure_ev': 0, 'white_balance': 'as shot',
                                 'post_lut': 'Disabled', 'highlight_recovery': False, 'gamut_compression': False},
                      'simulation': {'auto_exposure': False, 'grain': args.mode == 'film' and not args.no_grain,
                                     'grain_base_seed': args.grain_seed,
                                     'grain_seed_method': 'blake2b-64/base-seed+absolute-frame/v1',
                                     'grain_backend': 'seeded reference CPU', 'stochastic_glare': False},
                      'colour_note': 'sRGB display output; values outside [0,1] are clipped for video encoding. '
                                     'SDK gamut paths can differ in bright highlights; see docs/braw-processing.md.'}
            # Write the report before publishing the final movie, so serialization
            # or disk errors cannot leave an apparently successful movie alone.
            temporary_report = Path(directory) / 'report.json'
            temporary_report.write_text(json.dumps(report, indent=2) + '\n')
            temporary.replace(output)
            temporary_report.replace(report_path)
        print(f'Done: {output}\nReport: {report_path}')
        return 0
    except KeyboardInterrupt:
        print('\nCancelled; no partial movie was published.', file=sys.stderr)
        return 130
    except (ProcessingError, OSError, subprocess.SubprocessError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
