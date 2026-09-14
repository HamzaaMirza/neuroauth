"""Window index arithmetic -- the part that is easy to get wrong by one."""

import numpy as np
import pytest

from neuroauth.config import WindowConfig
from neuroauth.dsp.types import Recording
from neuroauth.dsp.windowing import (
    frame_signal,
    window_bounds,
    window_recording,
    window_stream_chunk,
)
from tests.synthetic import CH_NAMES, SFREQ


def test_window_count_at_fifty_percent_overlap(synthetic_recording: Recording) -> None:
    """60 s at 160 Hz, 2 s windows, 50% overlap yields 59 windows."""
    assert window_bounds(9600, 320, 160, drop_partial=True).size == 59
    windows = window_recording(synthetic_recording, WindowConfig())
    assert windows.data.shape == (59, 64, 320)


@pytest.mark.parametrize("n_samples", [320, 321, 479, 480, 481, 9600, 9760])
def test_no_window_runs_past_the_end(n_samples: int) -> None:
    """Every window fits, hops are uniform, and no further window would have fit."""
    starts = window_bounds(n_samples, 320, 160, drop_partial=True)
    assert starts[0] == 0
    assert np.all(np.diff(starts) == 160)
    assert np.all(starts + 320 <= n_samples)
    assert starts[-1] + 160 + 320 > n_samples


def test_partial_window_is_dropped_not_padded() -> None:
    """A trailing remainder shorter than a window is discarded, never zero-padded."""
    frames = frame_signal(np.ones((2, 1000)), 320, 160)
    assert frames.shape == (5, 2, 320)
    assert np.all(frames == 1.0)


def test_drop_partial_false_still_never_pads() -> None:
    """Documents current contract: the flag does not change output in either mode."""
    np.testing.assert_array_equal(
        window_bounds(1000, 320, 160, drop_partial=False),
        window_bounds(1000, 320, 160, drop_partial=True),
    )


def test_signal_shorter_than_one_window_yields_none() -> None:
    """Returns an empty index array and an empty frame stack rather than raising."""
    starts = window_bounds(100, 320, 160, drop_partial=True)
    assert starts.size == 0
    assert starts.dtype == np.int64
    assert frame_signal(np.zeros((3, 100)), 320, 160).shape == (0, 3, 320)


def test_frames_are_owned_not_views() -> None:
    """Mutating one window leaves its overlapping neighbour and the source untouched."""
    data = np.random.default_rng(0).normal(size=(2, 800))
    source = data.copy()
    frames = frame_signal(data, 320, 160)
    assert not np.shares_memory(frames, data)
    assert frames.flags.c_contiguous

    neighbour_head = frames[1, :, :160].copy()
    frames[0, :, 160:] = 0.0
    np.testing.assert_array_equal(frames[1, :, :160], neighbour_head)
    np.testing.assert_array_equal(data, source)


def test_onsets_match_frame_contents() -> None:
    """onsets_s[i] * sfreq indexes the first sample of window i in the source."""
    ramp = np.tile(np.arange(2000, dtype=np.float64), (3, 1))
    recording = Recording(ramp, SFREQ, ("A", "B", "C"), 1, 1, "eyes_open")
    windows = window_recording(recording, WindowConfig())
    np.testing.assert_array_equal(windows.data[:, 0, 0], windows.onsets_s * SFREQ)
    np.testing.assert_array_equal(np.diff(windows.data[:, 0, :], axis=1), 1.0)


def test_window_recording_carries_provenance(synthetic_recording: Recording) -> None:
    windows = window_recording(synthetic_recording, WindowConfig())
    assert (windows.subject_id, windows.run, windows.condition) == (1, 1, "eyes_open")
    assert windows.ch_names == synthetic_recording.ch_names
    assert windows.sfreq == SFREQ


@pytest.mark.parametrize("overlap", [-0.1, 1.0, 0.999])
def test_invalid_overlap_is_rejected(synthetic_recording: Recording, overlap: float) -> None:
    """Outside [0, 1) raises; 0.999 is in range but rounds to a zero-sample hop."""
    with pytest.raises(ValueError):
        window_recording(synthetic_recording, WindowConfig(overlap=overlap))


def test_stream_chunk_has_no_identity(synthetic_recording: Recording) -> None:
    """window_stream_chunk leaves subject_id, run, and condition as None."""
    windows = window_stream_chunk(
        synthetic_recording.data[:, :1600], SFREQ, CH_NAMES, WindowConfig()
    )
    assert (windows.subject_id, windows.run, windows.condition) == (None, None, None)
    assert windows.data.shape == (9, 64, 320)


def test_stream_chunk_onsets_are_monotonic_across_chunks(
    synthetic_recording: Recording,
) -> None:
    """onset_offset_s keeps successive chunks continuous in time."""
    chunk = 1600
    data = synthetic_recording.data
    first = window_stream_chunk(data[:, :chunk], SFREQ, CH_NAMES, WindowConfig())
    second = window_stream_chunk(
        data[:, chunk : 2 * chunk], SFREQ, CH_NAMES, WindowConfig(), onset_offset_s=chunk / SFREQ
    )
    onsets = np.concatenate([first.onsets_s, second.onsets_s])
    assert np.all(np.diff(onsets) > 0)
    assert second.onsets_s[0] == pytest.approx(10.0)


def test_stream_chunk_rejects_channel_mismatch(synthetic_recording: Recording) -> None:
    with pytest.raises(ValueError):
        window_stream_chunk(synthetic_recording.data, SFREQ, CH_NAMES[:32], WindowConfig())
