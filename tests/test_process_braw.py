"""Protocol and lifecycle checks; no SDK, media fixture, or FFmpeg required."""
from __future__ import annotations

import importlib.util
from io import BytesIO
from fractions import Fraction
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    'process_braw', Path(__file__).parents[1] / 'scripts/process_braw.py')
braw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(braw)


class FragmentedPipe(BytesIO):
    def read(self, size=-1):
        return super().read(min(size, 3))


def test_frame_reader_handles_fragmented_pipe_and_truncation():
    payload = struct.pack('<fff', 0.1, 0.2, 4.0)
    packet = struct.pack('<IIQ', 1, 1, 7) + payload
    frame = braw.read_frame(FragmentedPipe(packet))
    assert frame[:3] == (7, 1, 1)
    assert frame[3] == payload
    assert braw.read_frame(BytesIO()) is None
    for bad_packet in (packet[:10], packet[:-1], packet[:16]):
        with pytest.raises(braw.ProcessingError):
            braw.read_frame(FragmentedPipe(bad_packet))


def test_fractional_frame_rate_audio_cut_uses_sample_boundaries():
    # At 29.97 fps, frame 3 starts at sample 4804.8 and frame 5 at 8008.
    # Round the boundaries separately; rounding the duration would drift.
    args = SimpleNamespace(no_audio=False, start_frame=3)
    metadata = {'streams': [{'codec_type': 'audio', 'codec_name': 'pcm_s24le',
                             'sample_rate': '48000', 'start_time': '0'}]}
    _, descriptions, checks = braw.audio_options(args, metadata, Fraction(30000, 1001), 2, 100)
    assert descriptions[0]['start_sample'] == 4805
    assert descriptions[0]['end_sample'] == 8008
    assert checks[0][1] == 'atrim=start_sample=4805:end_sample=8008,asetpts=PTS-STARTPTS'


def test_existing_output_is_protected_before_loading_sdk(tmp_path, monkeypatch, capsys):
    source, output = tmp_path / 'input.braw', tmp_path / 'keep.mov'
    source.write_bytes(b'fixture')
    output.write_bytes(b'keep the existing movie')
    monkeypatch.setattr(braw.shutil, 'which', lambda name: name)
    def unexpected_build(*args):
        pytest.fail('SDK should not be loaded for an existing output')
    monkeypatch.setattr(braw, 'build_decoder', unexpected_build)
    assert braw.main([str(source), '-o', str(output)]) == 1
    assert output.read_bytes() == b'keep the existing movie'
    assert 'already exists' in capsys.readouterr().err


def test_keyboard_interrupt_terminates_owned_child():
    child = None
    with pytest.raises(KeyboardInterrupt):
        with braw.process([sys.executable, '-u', '-c',
                           'import time; print("ready"); time.sleep(60)'],
                          stdout=subprocess.PIPE) as child:
            assert child.stdout.readline() == b'ready\n'
            raise KeyboardInterrupt
    assert child.poll() is not None


def test_video_grain_defaults_and_absolute_frame_seeds():
    args = braw.parser().parse_args(['input.braw', '-o', 'output.mov'])
    assert args.no_grain is False
    assert args.grain_seed == 0
    frames = [braw.frame_grain_seed(0, index) for index in range(100)]
    assert len(set(frames)) == len(frames)
    # Rendering a cut or retrying a frame uses the absolute source index.
    assert [braw.frame_grain_seed(0, index) for index in range(20, 25)] == frames[20:25]
    assert braw.frame_grain_seed(1, 20) != frames[20]
