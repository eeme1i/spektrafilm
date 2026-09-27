"""Metal (MLX) version of the fused Poisson-binomial grain sampler.

Grain sampling is compute bound (a few dozen random draws per pixel), which
the Apple-silicon GPU does 5-15x faster than the numba CPU kernel. Every other
stage is memory bound and stays on the CPU; with unified memory the round
trip for one grain plane costs a few milliseconds.

MLX is an optional dependency (``pip install spektrafilm[gpu]``). When it is
missing, or no Metal device is present, :func:`available` returns False and
callers use the numba kernel in ``model.grain``.

The samplers follow ``fast_stats.poisson_sample`` / ``binomial_sample``
branch for branch, in float32, with a PCG32 stream per pixel instead of
numba's generator. Results are statistically equivalent, not bit-identical.
"""

from __future__ import annotations

import warnings

import numpy as np

_state = {'checked': False, 'kernel': None, 'mx': None}

_HEADER = """
struct Pcg32 { ulong state; ulong inc; };

inline uint pcg32_next(thread Pcg32& g) {
    ulong old = g.state;
    g.state = old * 6364136223846793005ul + g.inc;
    uint xorshifted = uint(((old >> 18u) ^ old) >> 27u);
    uint rot = uint(old >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31u));
}

// One independent stream per pixel: the pixel index selects the stream.
inline Pcg32 pcg32_seed(ulong seed, ulong stream) {
    Pcg32 g; g.state = 0ul; g.inc = (stream << 1u) | 1ul;
    pcg32_next(g); g.state += seed; pcg32_next(g);
    return g;
}

inline float next_uniform(thread Pcg32& g) {
    return float(pcg32_next(g) >> 8) * (1.0f / 16777216.0f);  // [0, 1)
}

inline float next_normal(thread Pcg32& g) {
    float u1 = max(next_uniform(g), 1e-7f);
    float u2 = next_uniform(g);
    return sqrt(-2.0f * log(u1)) * cos(6.28318530718f * u2);
}

inline int poisson_sample(float lam, thread Pcg32& g) {
    if (lam <= 0.0f) return 0;
    if (lam < 30.0f) {
        float limit = exp(-lam); float p = 1.0f; int k = 0;
        while (p > limit) { k++; p *= next_uniform(g); }
        return k - 1;
    }
    return max(0, int(round(lam + sqrt(lam) * next_normal(g))));
}

inline int binomial_sample(int n, float p, thread Pcg32& g) {
    if (p <= 0.0f) return 0;
    if (p >= 1.0f) return n;
    if (n < 25) {
        int count = 0;
        for (int k = 0; k < n; ++k) count += next_uniform(g) < p;
        return count;
    }
    float mean = n * p;
    float var = mean * (1.0f - p);
    if (var > 10.0f) return clamp(int(round(mean + sqrt(var) * next_normal(g))), 0, n);
    bool flip = p > 0.5f;
    float q = flip ? 1.0f - p : p;
    float u = next_uniform(g);
    float cdf = 0.0f; float prob = pow(1.0f - q, float(n)); int k = 0;
    while (cdf < u && k <= n) {
        cdf += prob;
        if (k < n) prob *= (float(n - k) / float(k + 1)) * (q / (1.0f - q));
        k++;
    }
    return flip ? n - (k - 1) : k - 1;
}
"""

# Mirrors model.grain._fast_poisson_binomial_grain.
_SOURCE = """
    uint idx = thread_position_in_grid.x;
    if (idx >= n_pixels[0]) return;
    float density_max = params[0];
    float n_particles_per_pixel = params[1];
    float grain_uniformity = params[2];
    Pcg32 g = pcg32_seed((ulong(seed[1]) << 32u) | ulong(seed[0]), ulong(idx));
    float od_particle = density_max / n_particles_per_pixel;
    float probability = min(max(float(density[idx]) / density_max, 1e-6f), 1.0f - 1e-6f);
    float saturation = 1.0f - probability * grain_uniformity * (1.0f - 1e-6f);
    int seeds = poisson_sample(n_particles_per_pixel / saturation, g);
    out[idx] = binomial_sample(seeds, probability, g) * od_particle * saturation;
"""


def _load():
    if not _state['checked']:
        _state['checked'] = True
        try:
            import mlx.core as mx
            if mx.metal.is_available():
                _state['mx'] = mx
                _state['kernel'] = mx.fast.metal_kernel(
                    name='spektrafilm_poisson_binomial_grain',
                    input_names=['density', 'params', 'seed', 'n_pixels'],
                    output_names=['out'],
                    source=_SOURCE,
                    header=_HEADER,
                )
        except ImportError:
            pass
    return _state['kernel']


def available() -> bool:
    """True when MLX is installed and a Metal GPU is usable."""
    return _load() is not None


def poisson_binomial_grain(density, density_max, n_particles_per_pixel, grain_uniformity,
                           seed=None):
    """GPU equivalent of ``model.grain._fast_poisson_binomial_grain``.

    Returns a float32 array shaped like ``density``, or None if the GPU path
    is unavailable or fails (the failure is reported once, then the GPU path
    stays off for the session).
    """
    kernel = _load()
    if kernel is None:
        return None
    mx = _state['mx']
    if seed is None:
        seed = np.random.randint(0, 2**63, dtype=np.int64)
    seed = int(seed) & 0xFFFFFFFFFFFFFFFF
    density = np.ascontiguousarray(density, dtype=np.float32)
    try:
        out = kernel(
            inputs=[
                mx.array(density.reshape(-1)),
                mx.array([density_max, n_particles_per_pixel, grain_uniformity], dtype=mx.float32),
                mx.array([seed & 0xFFFFFFFF, seed >> 32], dtype=mx.uint32),
                mx.array([density.size], dtype=mx.uint32),
            ],
            grid=(density.size, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(density.size,)],
            output_dtypes=[mx.float32],
        )[0]
        return np.asarray(out).reshape(density.shape)
    except Exception as exc:  # never let the accelerator break a render
        warnings.warn(f'GPU grain failed, falling back to CPU: {exc}', RuntimeWarning, stacklevel=2)
        _state['kernel'] = None
        return None


def warmup_gpu_grain() -> None:
    """Compile the Metal kernel (no-op without MLX)."""
    poisson_binomial_grain(np.full((4, 4), 1.0, dtype=np.float32), 2.2, 10.0, 0.98, seed=0)
