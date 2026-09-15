"""Bounded-context featurization: identical features for training, enrollment, and a live stream.

`sosfiltfilt` gets zero phase by filtering forward and then backward over its whole input.
An interior window of a 61 s recording sits seconds from both ends and never sees an edge
transient. The newest window of a live stream sits at the right edge, where the backward
pass starts. Past samples cannot help there, because the transient is on the side that has
no future samples yet. On S001R01, filtering each 2 s buffer on its own moved relative
delta power by 0.013 (median) and 0.058 (95th percentile) compared with whole-recording
filtering. That is the same `preprocess` function on both paths giving different features.
D-001 removed code skew; this is buffer skew.

The fix: every window's features are a pure function of one fixed raw slice, the window
plus ContextConfig margins on both sides, filtered with the Phase 1 `preprocess` and cropped
back to the window. Offline, the slice is cut from the recording. Live, a window is emitted
once its right margin has arrived. It is the same slice and the same function, so the two
paths agree bit for bit by construction, not just within a tolerance. `push_samples` is
tested for exactly that. Training and enrollment features are built here as well, never
with pipeline.process_recording, whose whole-recording filtering the live path cannot
reproduce.

The cost: a session's first decision arrives left + window + right = 6 s after its first
sample, and each later decision arrives right_margin_s = 2 s after its window ends. D-020
covers the causal-filter alternative and why it lost.

The only thing held here is raw samples waiting for their context, in memory. A StreamBuffer
is never persisted, logged, or sent to a client (hard rule 3).
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from neuroauth.config import PreprocessConfig, StreamingConfig
from neuroauth.dsp.types import Condition, FeatureMatrix, Recording

SETTLING_TOLERANCE: Final = 1e-3
"""Impulse-response envelope, relative to its peak, below which the filter counts as settled.

Fixed before the margins were chosen. The margins follow from it; they were not tuned on
data."""


def settling_time_s(
    config: PreprocessConfig,
    sfreq: float,
    *,
    tolerance: float = SETTLING_TOLERANCE,
) -> float:
    """Time after which the notch-plus-bandpass impulse response stays below tolerance.

    Data-free: computed from the filter design alone, by running a unit impulse through the
    same notch and bandpass sections `preprocess` uses, in a single forward pass. The
    backward pass of filtfilt is the same filter reversed in time, so this bounds how far
    an edge transient reaches into the signal from either end. At the defaults and 160 Hz
    the answer is 1.58 s.

    Args:
        config: Filter settings.
        sfreq: Hz.
        tolerance: Fraction of the peak absolute response.

    Returns:
        Seconds from the impulse to the last sample whose absolute response exceeds
        tolerance times the peak.

    Raises:
        ValueError: If tolerance is outside (0, 1) or any filter setting is invalid.
    """
    raise NotImplementedError("TODO(phase-2): impulse-response settling time")


def check_margins(streaming: StreamingConfig, sfreq: float) -> None:
    """Refuse context margins shorter than the filter's settling time.

    Called when a stream is configured and before any offline feature build, so a
    margin set too short fails loudly instead of showing up later as an unexplained drop
    in accuracy.

    Raises:
        ValueError: If either margin is below settling_time_s(streaming.pipeline.preprocess,
            sfreq).
    """
    raise NotImplementedError("TODO(phase-2): margin check against settling time")


def context_window_starts(
    n_samples: int,
    sfreq: float,
    streaming: StreamingConfig,
) -> NDArray[np.int64]:
    """Start indices of the windows whose full raw context fits inside the signal.

    Windows stay on the Phase 1 hop grid (starts at k * hop from sample 0, as in
    windowing.window_bounds), so a given start index names the same window on either path.
    The first start is the smallest grid point at or after the left margin. The last is the
    largest whose window plus right margin still ends by n_samples. A 61 s recording
    therefore yields about four fewer windows than in Phase 1.

    Args:
        n_samples: Signal length.
        sfreq: Hz.
        streaming: Window geometry and margins.

    Returns:
        (n_windows,) int64, ascending. Empty when the signal is shorter than one full
        context.

    Raises:
        ValueError: If n_samples is negative or the window geometry is invalid.
    """
    raise NotImplementedError("TODO(phase-2): context-aware window starts")


def featurize_windows(
    data: NDArray[np.float64],
    starts: NDArray[np.int64],
    sfreq: float,
    ch_names: tuple[str, ...],
    streaming: StreamingConfig,
    *,
    onset_offset_s: float = 0.0,
    subject_id: int | None = None,
    run: int | None = None,
    condition: Condition | None = None,
) -> FeatureMatrix:
    """Features for the windows at `starts`, each filtered over its own bounded raw context.

    For each start s: preprocess data[:, s - left : s + window + right], crop
    [left : left + window], then window and extract features with the Phase 1 functions.
    The result for a window depends only on those raw samples. That is the property that
    makes offline and live features identical.

    Pure: no I/O, and `data` is not mutated. Content never raises (as extract_features).

    Args:
        data: (n_channels, n_samples) raw volts, not yet filtered.
        starts: Window start indices whose full context lies inside data, e.g. from
            context_window_starts.
        sfreq: Hz.
        ch_names: Length n_channels.
        streaming: Filter, window, feature, quality settings and margins.
        onset_offset_s: Seconds before data[:, 0], so onsets stay monotonic across a
            stream.
        subject_id: Provenance for a known recording. None for a live stream.
        run: As subject_id.
        condition: As subject_id.

    Returns:
        A FeatureMatrix with one row per start, onsets_s = starts / sfreq + onset_offset_s.

    Raises:
        ValueError: If data is not 2-D, ch_names does not match, or any start's context
            falls outside data. Structural caller errors only.
    """
    raise NotImplementedError("TODO(phase-2): per-window bounded-context features")


def process_recording_bounded(recording: Recording, streaming: StreamingConfig) -> FeatureMatrix:
    """Offline Phase 2 features for one recording: every window with a full context.

    Replaces pipeline.process_recording for every Phase 2 purpose, including embedding
    fitting, enrollment, and evaluation. Equal to push_samples fed the whole recording in
    any chunking.

    Args:
        recording: Raw, as returned by load_recording. Not mutated.
        streaming: Full settings.

    Returns:
        A FeatureMatrix carrying the recording's provenance.
    """
    raise NotImplementedError("TODO(phase-2): offline bounded-context features")


@dataclass(frozen=True)
class StreamBuffer:
    """Raw samples held until the windows that need them have their full context.

    In memory only. Never persisted, logged, or sent to a client (hard rule 3).

    Attributes:
        samples: (n_channels, n_retained) raw volts. Holds only what a not-yet-emitted
            window's context still needs, so memory is bounded by one context length.
        first_index: Stream sample index of samples[:, 0], counted from segment start.
        next_start: Stream sample index of the next window to emit, on the hop grid.
        origin_s: Session time of the segment's first sample. It is 0.0 for a fresh
            stream, and the resume time after a gap reset, so onsets stay on session time.
    """

    samples: NDArray[np.float64]
    first_index: int
    next_start: int
    origin_s: float


def new_stream_buffer(
    n_channels: int, streaming: StreamingConfig, sfreq: float, *, origin_s: float = 0.0
) -> StreamBuffer:
    """An empty buffer for a new stream segment.

    A gap in the stream (a lost frame) starts a new segment: filtering across missing
    samples would splice two unrelated contexts. The session runtime owns that decision.

    Raises:
        ValueError: If n_channels is below 1, or check_margins fails.
    """
    raise NotImplementedError("TODO(phase-2): empty stream buffer")


def push_samples(
    buffer: StreamBuffer,
    chunk: NDArray[np.float64],
    sfreq: float,
    ch_names: tuple[str, ...],
    streaming: StreamingConfig,
) -> tuple[StreamBuffer, FeatureMatrix]:
    """Append raw samples and emit features for every window whose context is now complete.

    Chunk invariance is the contract: for any partition of a recording into chunks of any
    sizes (including 1 sample and chunks longer than a window), the rows emitted across
    all calls, concatenated, equal process_recording_bounded(recording) bit for bit in
    values, onsets_s, window_ok, channel_ok, and flags.

    Pure: returns a new buffer and never mutates the old one or the chunk.

    Args:
        buffer: State from the previous call or new_stream_buffer.
        chunk: (n_channels, n_new) raw volts. May be empty.
        sfreq: Hz.
        ch_names: Length n_channels.
        streaming: Must be the settings the buffer was created with.

    Returns:
        (buffer, features): the advanced buffer and a FeatureMatrix of the windows just
        completed, with zero rows when none were. subject_id, run, and condition are None.

    Raises:
        ValueError: If chunk is not 2-D or its channel axis does not match ch_names. The
            session runtime validates frames before calling, so the live path never hits
            this.
    """
    raise NotImplementedError("TODO(phase-2): chunk-invariant stream buffering")
