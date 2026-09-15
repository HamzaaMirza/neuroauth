"""Bounded-context featurization: offline and live features identical, edges bounded (D-020)."""

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from neuroauth.config import ContextConfig, StreamingConfig
from neuroauth.dsp.features import extract_features
from neuroauth.dsp.io import load_recording
from neuroauth.dsp.preprocess import preprocess
from neuroauth.dsp.streaming import (
    check_margins,
    context_window_starts,
    featurize_windows,
    new_stream_buffer,
    process_recording_bounded,
    push_samples,
    settling_time_s,
    stack_feature_rows,
)
from neuroauth.dsp.types import FeatureMatrix, Recording, WindowSet
from tests.synthetic import CH_NAMES, SFREQ, eeg_like_data

EDGE_TOLERANCE = 5e-3
"""Max |difference| in relative band power against whole-recording filtering.

This judges a unit test, not a result, so D-016's a priori rule does not govern it. It was
chosen after the S001R01 measurement (D-020) and is disclosed as such."""

STREAMING = StreamingConfig()
CHANNELS = CH_NAMES[:8]
WINDOW = 320
CONTEXT = round(6.0 * SFREQ)
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
_S001 = DATA_DIR / "MNE-eegbci-data" / "files" / "eegmmidb" / "1.0.0" / "S001"
needs_data = pytest.mark.skipif(not _S001.exists(), reason="eegmmidb S001 is not downloaded")


def _recording(data: NDArray[np.float64]) -> Recording:
    return Recording(
        data=data,
        sfreq=SFREQ,
        ch_names=CHANNELS,
        subject_id=1,
        run=1,
        condition="eyes_open",
    )


def _chunk_sizes(n_samples: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    sizes: list[int] = []
    while sum(sizes) < n_samples:
        sizes.append(int(rng.choice([1, 2, 17, 160, 319, 320, 321, 777])))
    return sizes


def _stream(data: NDArray[np.float64], sizes: list[int]) -> FeatureMatrix:
    buffer = new_stream_buffer(data.shape[0], STREAMING, SFREQ)
    pieces: list[FeatureMatrix] = []
    position = 0
    for size in sizes:
        buffer, features = push_samples(
            buffer, data[:, position : position + size], SFREQ, CHANNELS, STREAMING
        )
        pieces.append(features)
        position += size
    return stack_feature_rows(pieces)


def _interior_starts(n_samples: int) -> NDArray[np.int64]:
    """Context windows at least 6 s from both recording ends, so the whole-recording
    reference is itself free of edge effects."""
    starts = context_window_starts(n_samples, SFREQ, STREAMING)
    guard = round(6.0 * SFREQ)
    return starts[(starts >= guard) & (starts + WINDOW <= n_samples - guard)]


def _window_features(
    filtered: list[NDArray[np.float64]], ch_names: tuple[str, ...]
) -> NDArray[np.float64]:
    window_set = WindowSet(
        data=np.stack(filtered),
        sfreq=SFREQ,
        ch_names=ch_names,
        onsets_s=np.zeros(len(filtered)),
        subject_id=None,
        run=None,
        condition=None,
    )
    return extract_features(
        window_set, STREAMING.pipeline.features, STREAMING.pipeline.quality
    ).values


def _edge_errors(data: NDArray[np.float64], ch_names: tuple[str, ...]) -> tuple[float, float]:
    """(bounded-context max error, naive per-buffer max error) against whole-recording filtering."""
    config = STREAMING.pipeline.preprocess
    starts = _interior_starts(data.shape[1])
    assert starts.size >= 10
    whole = preprocess(data, SFREQ, config)
    reference = _window_features([whole[:, s : s + WINDOW] for s in starts], ch_names)
    bounded = featurize_windows(data, starts, SFREQ, ch_names, STREAMING).values
    naive = _window_features(
        [preprocess(data[:, s : s + WINDOW], SFREQ, config) for s in starts], ch_names
    )
    return float(np.abs(bounded - reference).max()), float(np.abs(naive - reference).max())


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_stream_matches_offline_bit_for_bit(seed: int) -> None:
    """Any chunking of the live stream reproduces the offline training features exactly."""
    data = eeg_like_data(seed=seed, duration_s=30.0)
    data[3, 1500:1520] = 1e-3  # a clipping burst, so quality flags are exercised too
    offline = process_recording_bounded(_recording(data), STREAMING)
    live = _stream(data, _chunk_sizes(data.shape[1], seed))

    assert offline.values.shape[0] > 0
    assert not offline.quality.window_ok.all() or any(offline.quality.flags)
    assert live.feature_names == offline.feature_names
    np.testing.assert_array_equal(live.values, offline.values)
    np.testing.assert_array_equal(live.onsets_s, offline.onsets_s)
    np.testing.assert_array_equal(live.quality.window_ok, offline.quality.window_ok)
    np.testing.assert_array_equal(live.quality.channel_ok, offline.quality.channel_ok)
    assert live.quality.flags == offline.quality.flags


def test_bounded_context_matches_whole_recording_filtering() -> None:
    """With settled margins, edge transients stay below the tolerance."""
    bounded_error, _ = _edge_errors(eeg_like_data(seed=3, duration_s=40.0), CHANNELS)
    assert bounded_error <= EDGE_TOLERANCE


def test_naive_per_buffer_filtering_fails_the_same_check() -> None:
    """Negative control: filtering each 2 s buffer alone breaks the same tolerance, so the
    check above has teeth."""
    _, naive_error = _edge_errors(eeg_like_data(seed=3, duration_s=40.0), CHANNELS)
    assert naive_error > EDGE_TOLERANCE


@pytest.mark.parametrize(
    "context", [ContextConfig(left_margin_s=1.0), ContextConfig(right_margin_s=1.0)]
)
def test_margins_below_settling_time_are_refused(context: ContextConfig) -> None:
    streaming = StreamingConfig(context=context)
    with pytest.raises(ValueError, match="settling"):
        check_margins(streaming, SFREQ)
    with pytest.raises(ValueError, match="settling"):
        process_recording_bounded(_recording(eeg_like_data(duration_s=10.0)), streaming)
    with pytest.raises(ValueError, match="settling"):
        new_stream_buffer(len(CHANNELS), streaming, SFREQ)


def test_settling_time_at_the_defaults() -> None:
    """Data-free, from the filter design: the figure D-020's margins are derived from."""
    assert settling_time_s(STREAMING.pipeline.preprocess, SFREQ) == pytest.approx(1.58, abs=0.01)


def test_context_window_starts_stay_on_the_hop_grid() -> None:
    np.testing.assert_array_equal(
        context_window_starts(round(10.0 * SFREQ), SFREQ, STREAMING), [320, 480, 640, 800, 960]
    )
    assert context_window_starts(CONTEXT - 1, SFREQ, STREAMING).size == 0


def test_first_window_waits_for_its_full_context() -> None:
    """Warm-up: nothing is emitted until left margin + window + right margin have arrived."""
    data = eeg_like_data(duration_s=10.0)
    buffer = new_stream_buffer(len(CHANNELS), STREAMING, SFREQ)
    buffer, features = push_samples(buffer, data[:, : CONTEXT - 1], SFREQ, CHANNELS, STREAMING)
    assert features.values.shape[0] == 0
    buffer, features = push_samples(
        buffer, data[:, CONTEXT - 1 : CONTEXT], SFREQ, CHANNELS, STREAMING
    )
    np.testing.assert_array_equal(features.onsets_s, [2.0])


def test_buffer_memory_is_bounded() -> None:
    data = eeg_like_data(duration_s=30.0)
    buffer = new_stream_buffer(len(CHANNELS), STREAMING, SFREQ)
    largest = 0
    for position in range(0, data.shape[1], 50):
        buffer, _ = push_samples(
            buffer, data[:, position : position + 50], SFREQ, CHANNELS, STREAMING
        )
        largest = max(largest, buffer.samples.shape[1])
    assert largest <= CONTEXT + 160


def test_push_does_not_mutate_its_inputs() -> None:
    data = eeg_like_data(duration_s=10.0)
    buffer, _ = push_samples(
        new_stream_buffer(len(CHANNELS), STREAMING, SFREQ),
        data[:, :500],
        SFREQ,
        CHANNELS,
        STREAMING,
    )
    samples_before = buffer.samples.copy()
    chunk = data[:, 500:900].copy()
    push_samples(buffer, chunk, SFREQ, CHANNELS, STREAMING)
    np.testing.assert_array_equal(buffer.samples, samples_before)
    np.testing.assert_array_equal(chunk, data[:, 500:900])


def test_featurize_rejects_a_context_outside_the_data() -> None:
    data = eeg_like_data(duration_s=10.0)
    with pytest.raises(ValueError, match="context"):
        featurize_windows(data, np.array([160]), SFREQ, CHANNELS, STREAMING)


@pytest.mark.slow
@needs_data
def test_s001r01_bounded_context_within_tolerance_and_naive_outside() -> None:
    """The synthetic checks, repeated on the recording the D-020 measurement used."""
    recording = load_recording(1, 1, DATA_DIR)
    bounded_error, naive_error = _edge_errors(recording.data, recording.ch_names)
    assert bounded_error <= EDGE_TOLERANCE < naive_error
