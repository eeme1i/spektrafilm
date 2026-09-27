from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from spektrafilm_gui import controller_runtime as runtime_module


class FakeSignal:
    def __init__(self) -> None:
        self.emitted: list[object] = []

    def emit(self, value) -> None:
        self.emitted.append(value)


def test_execute_simulation_request_uses_runtime_runner_without_padding() -> None:
    request = runtime_module.SimulationRequest(
        mode_label='Preview',
        image=np.full((2, 2, 3), 0.25, dtype=np.float32),
        params=object(),
        output_color_space='ACES2065-1',
        use_display_transform=True,
    )
    captured: dict[str, object] = {}

    result = runtime_module.execute_simulation_request(
        request,
        run_simulation_fn=lambda image, params, on_progress=None: np.full((4, 4, 3), 0.5, dtype=np.float32),
        prepare_output_display_image_fn=lambda image, **kwargs: _capture_preview_result(captured, image, **kwargs),
    )

    np.testing.assert_allclose(captured['display_args']['image'], np.full((4, 4, 3), 0.5, dtype=np.float32))
    assert result.mode_label == 'Preview'
    np.testing.assert_allclose(result.float_image, np.full((4, 4, 3), 0.5, dtype=np.float32))
    assert result.status_message == 'Display transform: active'


def test_simulation_worker_emits_failure_message() -> None:
    request = runtime_module.SimulationRequest(
        mode_label='Preview',
        image=np.zeros((1, 1, 3), dtype=np.float32),
        params=object(),
        output_color_space='sRGB',
        use_display_transform=False,
    )
    worker = runtime_module.SimulationWorker(
        request,
        execute_request=lambda request, on_progress: (_ for _ in ()).throw(ValueError('bad simulation')),
    )
    worker.signals = SimpleNamespace(finished=FakeSignal(), failed=FakeSignal(), progress=FakeSignal())

    worker.run()

    assert worker.signals.finished.emitted == []
    assert worker.signals.failed.emitted == ['ValueError: bad simulation']


def test_prepare_input_color_preview_image_converts_to_srgb_float_preview() -> None:
    captured: dict[str, object] = {}

    def fake_rgb_to_rgb(image, input_color_space, output_color_space, apply_cctf_decoding, apply_cctf_encoding):
        captured['call'] = {
            'image': image.copy(),
            'input_color_space': input_color_space,
            'output_color_space': output_color_space,
            'apply_cctf_decoding': apply_cctf_decoding,
            'apply_cctf_encoding': apply_cctf_encoding,
        }
        return np.full((1, 1, 3), 0.5, dtype=np.float32)

    preview = runtime_module.prepare_input_color_preview_image(
        np.full((1, 1, 3), 0.25, dtype=np.float32),
        input_color_space='Display P3',
        apply_cctf_decoding=True,
        colour_module=SimpleNamespace(RGB_to_RGB=fake_rgb_to_rgb),
    )

    assert preview.dtype == np.float32
    np.testing.assert_allclose(preview, np.full((1, 1, 3), 0.5, dtype=np.float32))
    assert captured['call']['input_color_space'] == 'Display P3'
    assert captured['call']['output_color_space'] == runtime_module.DISPLAY_PREVIEW_COLOR_SPACE
    assert captured['call']['apply_cctf_decoding'] is True
    assert captured['call']['apply_cctf_encoding'] is True


def _capture_preview_result(captured: dict[str, object], image: np.ndarray, **kwargs):
    captured['display_args'] = {'image': image.copy(), **kwargs}
    return np.full((6, 6, 3), 99, dtype=np.uint8), 'Display transform: active'

class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_execute_simulation_request_reports_setup_stages_and_display() -> None:
    from spektrafilm.runtime import StageProgress

    request = runtime_module.SimulationRequest(
        mode_label='Scan',
        image=np.zeros((2, 2, 3), dtype=np.float32),
        params=object(),
        output_color_space='sRGB',
        use_display_transform=False,
    )
    clock = FakeClock()
    steps: list[runtime_module.ProgressStep] = []

    def run_simulation(image, params, on_progress=None):
        clock.now = 1.5  # setup took 1.5 s
        for index, label in enumerate(('filming.develop', 'scanning.scan_print')):
            on_progress(StageProgress(label, index, 2, finished=False))
            on_progress(StageProgress(label, index, 2, finished=True, elapsed=2.0))
        return image

    def prepare_display(image, **kwargs):
        clock.now += 0.25
        return image, 'Display transform: disabled'

    runtime_module.execute_simulation_request(
        request,
        run_simulation_fn=run_simulation,
        prepare_output_display_image_fn=prepare_display,
        on_progress=steps.append,
        clock=clock,
    )

    assert [(s.label, s.finished, s.index, s.count) for s in steps] == [
        ('setup', False, 0, None),
        ('setup', True, 0, 4),
        ('filming.develop', False, 1, 4),
        ('filming.develop', True, 1, 4),
        ('scanning.scan_print', False, 2, 4),
        ('scanning.scan_print', True, 2, 4),
        ('display', False, 3, 4),
        ('display', True, 3, 4),
    ]
    assert steps[1].elapsed == 1.5
    assert steps[-1].elapsed == 0.25


def test_simulation_progress_messages() -> None:
    progress = runtime_module.SimulationProgress('Scan', started_at=10.0)
    step = runtime_module.ProgressStep

    assert progress.running_message(10.5) == 'Computing scan... 0.5 s'

    progress.update(step('setup', finished=False, index=0), now=10.0)
    assert progress.running_message(10.4) == 'Scan: Setting up · 0.4 s · 0.4 s total'

    progress.update(step('setup', finished=True, elapsed=1.0, index=0, count=4), now=11.0)
    progress.update(step('filming.develop', finished=False, index=1, count=4), now=11.0)
    assert progress.running_message(13.0) == 'Scan: Developing film (2/4) · 2.0 s · 3.0 s total'

    progress.update(step('filming.develop', finished=True, elapsed=6.0, index=1, count=4), now=17.0)
    progress.update(step('display', finished=False, index=3, count=4), now=17.0)
    progress.update(step('display', finished=True, elapsed=0.1, index=3, count=4), now=17.1)
    # display is under 2 % of the total, so it is left out of the summary
    assert progress.summary_message(20.0) == (
        'Scan completed in 10.0 s · Setting up 1.0 s (10%) · Developing film 6.0 s (60%)'
    )
