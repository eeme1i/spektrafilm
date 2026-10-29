"""Measure RAW import and resolution scaling with GUI-equivalent fast settings."""
from __future__ import annotations

import argparse
import json
import platform
from collections import Counter
from pathlib import Path
from time import perf_counter

import numba
import numpy as np

from spektrafilm import Simulator, digest_params, init_params
from spektrafilm.runtime import pipeline as pipeline_module
from spektrafilm.runtime.topology import plan_topology
from spektrafilm.runtime.topology import Node
from spektrafilm.runtime.stages import scanning as scanning_module
from spektrafilm.model import grain
from spektrafilm.utils import gpu_grain
from spektrafilm.utils.crop_resize import rescale_nearest_antialiased
from spektrafilm.utils.numba_warmup import warmup
from spektrafilm.utils.raw_file_processor import load_and_process_raw_file


def release_consumed_taps(topology, inject, collect, image, *, on_start=None, on_fire=None):
    """Benchmark-only dispatcher experiment; retain values through their last reader."""
    plan = plan_topology(topology, inject, collect)
    remaining = Counter(tap for node in plan for tap in node.reads)
    state = {inject: image}
    for index, node in enumerate(plan):
        if on_start:
            on_start(node, index, len(plan))
        start = perf_counter()
        node.fire(state)
        elapsed = perf_counter()-start
        for tap in node.reads:
            remaining[tap] -= 1
        for tap in list(state):
            if tap != collect and remaining[tap] == 0:
                del state[tap]
        if on_fire:
            on_fire(node, elapsed)
    return state[collect]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw', type=Path)
    parser.add_argument('--film', default='kodak_portra_400')
    parser.add_argument('--print', dest='paper', default='kodak_portra_endura')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--release-taps', action='store_true', help='Try freeing consumed intermediate images')
    parser.add_argument('--float32-input', action='store_true', help='Try keeping the imported image as float32')
    parser.add_argument('--chunk-gamut', type=int, default=0, help='Experiment with gamut conversion in row blocks')
    parser.add_argument('--scales', nargs='+', type=float, default=[0.5, 1.0])
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--out', type=Path, default=Path('/tmp/spektrafilm-raw-scaling.json'))
    args = parser.parse_args()
    params = init_params(args.film, args.paper)
    params.settings.use_fast_stats = True
    params.settings.use_enlarger_lut = True
    params.settings.use_scanner_lut = True
    params.settings.use_gpu = not args.cpu
    params = digest_params(params)
    report = {'raw': str(args.raw), 'film': args.film, 'print': args.paper, 'machine': platform.machine(),
              'numba_threads': numba.get_num_threads(), 'gpu_requested': not args.cpu,
              'release_taps_experiment': args.release_taps, 'float32_input_experiment': args.float32_input,
              'chunk_gamut_rows_experiment': args.chunk_gamut,
              'output_gamut_algorithm': params.io.output_gamut_compress.algorithm,
              'gpu_available': gpu_grain.available(), 'runs': []}
    print(json.dumps({k:v for k,v in report.items() if k != 'runs'}), flush=True)
    start = perf_counter()
    image = load_and_process_raw_file(args.raw, output_colorspace=params.io.input_color_space,
                                     output_cctf_encoding=params.io.input_cctf_decoding)
    report['raw_import_seconds'] = perf_counter()-start
    report['input_shape'] = list(image.shape)
    report['import_dtype'] = str(image.dtype)
    print('RAW import', report['raw_import_seconds'], image.shape, image.dtype, flush=True)
    if args.float32_input:
        image = image.astype(np.float32)
    report['retained_input_bytes'] = image.nbytes
    if args.release_taps or args.chunk_gamut:
        # Compare deterministic output before benchmarking the memory-only change.
        import copy
        deterministic = copy.deepcopy(params)
        deterministic.debug.lut_mode = True
        deterministic = digest_params(deterministic)
        # At the tested 128-row block size this patch crosses a block boundary.
        patch = image[::16, ::16].copy()
        expected = Simulator(deterministic).process(patch)
        if args.release_taps:
            pipeline_module.run_topology = release_consumed_taps
        if args.chunk_gamut:
            original_compress = scanning_module.compress_rgb
            def chunked_compress(rgb, *positional, **kwargs):
                output = np.empty_like(rgb)
                for start in range(0, rgb.shape[0], args.chunk_gamut):
                    stop = start+args.chunk_gamut
                    output[start:stop] = original_compress(rgb[start:stop], *positional, **kwargs)
                return output
            scanning_module.compress_rgb = chunked_compress
        actual = Simulator(deterministic).process(patch)
        np.testing.assert_array_equal(actual, expected)
        report['experiments_deterministic_output_identical'] = True
        report['deterministic_validation_shape'] = list(patch.shape)
        print('Experiments match deterministic reference exactly:', patch.shape, flush=True)
    start = perf_counter()
    warmup()
    report['warmup_seconds'] = perf_counter()-start
    print('Warmup', report['warmup_seconds'], flush=True)
    original_sampler = grain.layer_particle_model
    original_fire = Node.fire
    buffers = []
    def observed_fire(node, state):
        original_fire(node, state)
        buffers.append({'stage': node.label, 'buffers': {
            tap: {'bytes': value.nbytes, 'dtype': str(value.dtype)}
            for tap, value in state.items() if isinstance(value, np.ndarray)}})
    Node.fire = observed_fire
    samples = []
    def timed_sampler(density, *positional, **kwargs):
        start = perf_counter()
        result = original_sampler(density, *positional, **kwargs)
        samples.append({'seconds': perf_counter()-start,
                        'density_max': float(kwargs['density_max']),
                        'particles_per_pixel': float(kwargs['n_particles_per_pixel']),
                        'mean_development_probability': float(np.mean(density)/kwargs['density_max'])})
        return result
    grain.layer_particle_model = timed_sampler
    try:
        for scale in args.scales:
            target = image if scale == 1 else rescale_nearest_antialiased(image, scale)
            if args.float32_input:
                target = target.astype(np.float32, copy=False)
            simulator = Simulator(params)
            for repeat in range(args.repeats):
                samples.clear()
                buffers.clear()
                def progress(event):
                    if event.finished:
                        print(f'{scale:g}x #{repeat+1}: {event.label}: {event.elapsed:.3f}s', flush=True)
                result = simulator.process(target, on_progress=progress)
                run = {'scale': scale, 'repeat': repeat+1, 'shape': list(target.shape),
                       'megapixels': target.shape[0]*target.shape[1]/1e6,
                       'seconds': simulator.get_total_elapsed_time(),
                       'timings': dict(simulator.get_timings()), 'grain_layers': list(samples),
                       'retained_buffers_after_node_before_release': list(buffers)}
                report['runs'].append(run)
                args.out.write_text(json.dumps(report, indent=2)+'\n')
                print('TOTAL', run['seconds'], flush=True)
                del result
            del simulator, target
    finally:
        grain.layer_particle_model = original_sampler
        Node.fire = original_fire
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(args.out, flush=True)


if __name__ == '__main__':
    main()
