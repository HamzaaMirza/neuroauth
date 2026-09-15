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

Each window is filtered and featurized on its own, in an array of the same shape on both
paths, so no batching difference can change a single bit.

The cost: a session's first decision arrives left + window + right = 6 s after its first
sample, and each later decision arrives right_margin_s = 2 s after its window ends (D-020).

The only thing held here is raw samples waiting for their context, in memory. A StreamBuffer
is never persisted, logged, or sent to a client (hard rule 3).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, iirnotch, sosfilt, tf2sos

from neuroauth.config import PreprocessConfig, StreamingConfig
from neuroauth.dsp.features import extract_features
from neuroauth.dsp.preprocess import preprocess
from neuroauth.dsp.types import Condition, FeatureMatrix, QualityReport, Recording, WindowSet
from neuroauth.dsp.windowing import _window_geometry

SETTLING_TOLERANCE: Final = 1e-3
"""Impulse-response envelope, relative to its peak, below which the filter counts as settled.

Data-free. The margins follow from it rather than being tuned on data (D-020)."""

_MIN_IMPULSE_S: Final = 30.0


def _filter_sections(config: PreprocessConfig, sfreq: float) -> NDArray[np.float64]:
    """The notch and bandpass sections `preprocess` applies, validated the same way."""
    nyquist = sfreq / 2.0
    if sfreq <= 0.0:
        raise ValueError(f"sfreq must be positive, got {sfreq}")
    if not 0.0 < config.bandpass_low < config.bandpass_high < nyquist:
        raise ValueError(
            f"need 0 < bandpass_low < bandpass_high < Nyquist ({nyquist} Hz), got "
            f"{config.bandpass_low}-{config.bandpass_high} Hz"
        )
    if config.bandpass_order < 1:
        raise ValueError(f"bandpass_order must be at least 1, got {config.bandpass_order}")
    if config.notch_freq <= 0.0 or config.notch_q <= 0.0 or config.notch_harmonics < 1:
        raise ValueError("notch_freq and notch_q must be positive and notch_harmonics >= 1")

    sections: list[NDArray[np.float64]] = []
    for harmonic in range(1, config.notch_harmonics + 1):
        target_hz = harmonic * config.notch_freq
        if target_hz >= nyquist:
            break
        sections.append(tf2sos(*iirnotch(target_hz, config.notch_q, fs=sfreq)))
    sections.append(
        butter(
            config.bandpass_order,
            [config.bandpass_low, config.bandpass_high],
            btype="bandpass",
            fs=sfreq,
            output="sos",
        )
    )
    return np.vstack(sections)


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
        ValueError: If tolerance is outside (0, 1), any filter setting is invalid, or the
            response has not settled within the simulated duration.
    """
    if not 0.0 < tolerance < 1.0:
        raise ValueError(f"tolerance must be in (0, 1), got {tolerance}")
    sos = _filter_sections(config, sfreq)
    n_samples = round(max(_MIN_IMPULSE_S, 100.0 / config.bandpass_low) * sfreq)
    impulse = np.zeros(n_samples, dtype=np.float64)
    impulse[0] = 1.0
    envelope = np.abs(sosfilt(sos, impulse))
    last = int(np.flatnonzero(envelope > tolerance * envelope.max()).max())
    if last >= n_samples - round(sfreq):
        raise ValueError("the impulse response did not settle within the simulated duration")
    return last / sfreq


def _geometry(sfreq: float, streaming: StreamingConfig) -> tuple[int, int, int, int]:
    """(window, hop, left, right) in samples."""
    window, hop = _window_geometry(sfreq, streaming.pipeline.window)
    context = streaming.context
    if context.left_margin_s < 0.0 or context.right_margin_s < 0.0:
        raise ValueError("context margins must be non-negative")
    return window, hop, round(context.left_margin_s * sfreq), round(context.right_margin_s * sfreq)


def check_margins(streaming: StreamingConfig, sfreq: float) -> None:
    """Refuse context margins shorter than the filter's settling time.

    Called when a stream is configured and before any offline feature build, so a
    margin set too short fails loudly instead of showing up later as an unexplained drop
    in accuracy.

    Raises:
        ValueError: If either margin is below settling_time_s(streaming.pipeline.preprocess,
            sfreq).
    """
    settle = settling_time_s(streaming.pipeline.preprocess, sfreq)
    context = streaming.context
    for name, margin in (
        ("left_margin_s", context.left_margin_s),
        ("right_margin_s", context.right_margin_s),
    ):
        if margin < settle:
            raise ValueError(
                f"{name} = {margin} s is shorter than the filter settling time "
                f"({settle:.2f} s at tolerance {SETTLING_TOLERANCE:g}); see D-020"
            )


def _first_start(left: int, hop: int) -> int:
    return -(-left // hop) * hop


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
    if n_samples < 0:
        raise ValueError(f"n_samples must be non-negative, got {n_samples}")
    window, hop, left, right = _geometry(sfreq, streaming)
    first = _first_start(left, hop)
    last = n_samples - window - right
    if last < first:
        return np.empty(0, dtype=np.int64)
    return np.arange(first, last + 1, hop, dtype=np.int64)


def stack_feature_rows(pieces: Sequence[FeatureMatrix]) -> FeatureMatrix:
    """Concatenate FeatureMatrices row-wise, e.g. the outputs of successive push_samples calls.

    Provenance is taken from the first piece. Pieces must share feature_names and provenance.

    Raises:
        ValueError: If pieces is empty, or feature_names or provenance differ.
    """
    if not pieces:
        raise ValueError("no feature matrices to stack")
    first = pieces[0]
    for piece in pieces[1:]:
        if piece.feature_names != first.feature_names:
            raise ValueError("feature matrices with different feature_names cannot be stacked")
        if (piece.subject_id, piece.run, piece.condition) != (
            first.subject_id,
            first.run,
            first.condition,
        ):
            raise ValueError("feature matrices with different provenance cannot be stacked")
    return FeatureMatrix(
        values=np.concatenate([piece.values for piece in pieces], axis=0),
        feature_names=first.feature_names,
        quality=QualityReport(
            window_ok=np.concatenate([piece.quality.window_ok for piece in pieces]),
            channel_ok=np.concatenate([piece.quality.channel_ok for piece in pieces], axis=0),
            flags=tuple(flag for piece in pieces for flag in piece.quality.flags),
        ),
        onsets_s=np.concatenate([piece.onsets_s for piece in pieces]),
        subject_id=first.subject_id,
        run=first.run,
        condition=first.condition,
    )


def featurize_windows(
    data: NDArray[np.float64],
    starts: NDArray[np.int64],
    sfreq: float,
    ch_names: tuple[str, ...],
    streaming: StreamingConfig,
    *,
    onset_offset_s: float = 0.0,
    index_offset: int = 0,
    subject_id: int | None = None,
    run: int | None = None,
    condition: Condition | None = None,
) -> FeatureMatrix:
    """Features for the windows at `starts`, each filtered over its own bounded raw context.

    For each start s: preprocess data[:, s - left : s + window + right], crop
    [left : left + window], then extract features with the Phase 1 function. Every window is
    processed alone, in arrays of identical shape, so the result for a window depends only on
    those raw samples. That is the property that makes offline and live features identical.

    Pure: no I/O, and `data` is not mutated. Content never raises (as extract_features).

    Args:
        data: (n_channels, n_samples) raw volts, not yet filtered.
        starts: Window start indices into data whose full context lies inside data, e.g.
            from context_window_starts.
        sfreq: Hz.
        ch_names: Length n_channels.
        streaming: Filter, window, feature, quality settings and margins.
        onset_offset_s: Seconds added to every onset (session time of stream index 0).
        index_offset: Stream index of data[:, 0]. Onsets are (start + index_offset) / sfreq +
            onset_offset_s, computed from integers so they match the offline path exactly.
        subject_id: Provenance for a known recording. None for a live stream.
        run: As subject_id.
        condition: As subject_id.

    Returns:
        A FeatureMatrix with one row per start.

    Raises:
        ValueError: If data is not 2-D, ch_names does not match, starts is not 1-D, or any
            start's context falls outside data. Structural caller errors only.
    """
    x = np.asarray(data, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected (n_channels, n_samples), got shape {x.shape}")
    if x.shape[0] != len(ch_names):
        raise ValueError(f"data has {x.shape[0]} channels but {len(ch_names)} ch_names")
    indices = np.asarray(starts, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError(f"starts must be 1-D, got shape {indices.shape}")
    window, _, left, right = _geometry(sfreq, streaming)
    if indices.size and (
        int(indices.min()) - left < 0 or int(indices.max()) + window + right > x.shape[1]
    ):
        raise ValueError("a window's context falls outside the data")

    config = streaming.pipeline

    def one(data_3d: NDArray[np.float64], onsets: NDArray[np.float64]) -> FeatureMatrix:
        window_set = WindowSet(
            data=data_3d,
            sfreq=sfreq,
            ch_names=ch_names,
            onsets_s=onsets,
            subject_id=subject_id,
            run=run,
            condition=condition,
        )
        return extract_features(window_set, config.features, config.quality)

    if indices.size == 0:
        return one(np.empty((0, x.shape[0], window), dtype=np.float64), np.empty(0))

    pieces: list[FeatureMatrix] = []
    for start in indices.tolist():
        context = np.ascontiguousarray(x[:, start - left : start + window + right])
        clean = preprocess(context, sfreq, config.preprocess)[:, left : left + window]
        onset = np.array([(start + index_offset) / sfreq + onset_offset_s], dtype=np.float64)
        pieces.append(one(np.ascontiguousarray(clean)[None, :, :], onset))
    return stack_feature_rows(pieces)


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

    Raises:
        ValueError: If the margins are shorter than the settling time.
    """
    check_margins(streaming, recording.sfreq)
    starts = context_window_starts(recording.data.shape[1], recording.sfreq, streaming)
    return featurize_windows(
        recording.data,
        starts,
        recording.sfreq,
        recording.ch_names,
        streaming,
        subject_id=recording.subject_id,
        run=recording.run,
        condition=recording.condition,
    )


@dataclass(frozen=True)
class StreamBuffer:
    """Raw samples held until the windows that need them have their full context.

    In memory only. Never persisted, logged, or sent to a client (hard rule 3).

    Attributes:
        samples: (n_channels, n_retained) raw volts. Holds only what a not-yet-emitted
            window's context still needs, so memory is bounded by one context plus a hop.
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
    if n_channels < 1:
        raise ValueError(f"n_channels must be at least 1, got {n_channels}")
    check_margins(streaming, sfreq)
    _, hop, left, _ = _geometry(sfreq, streaming)
    return StreamBuffer(
        samples=np.empty((n_channels, 0), dtype=np.float64),
        first_index=0,
        next_start=_first_start(left, hop),
        origin_s=origin_s,
    )


def segment_warming_up(buffer: StreamBuffer, sfreq: float, streaming: StreamingConfig) -> bool:
    """True until the current segment has emitted its first window."""
    _, hop, left, _ = _geometry(sfreq, streaming)
    return buffer.next_start == _first_start(left, hop)


def samples_until_next_window(
    buffer: StreamBuffer, sfreq: float, streaming: StreamingConfig
) -> int:
    """Raw samples still needed before the next window's full context has arrived."""
    window, _, _, right = _geometry(sfreq, streaming)
    have = buffer.first_index + int(buffer.samples.shape[1])
    return max(0, buffer.next_start + window + right - have)


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
    x = np.asarray(chunk, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected (n_channels, n_samples), got shape {x.shape}")
    if x.shape[0] != len(ch_names) or buffer.samples.shape[0] != len(ch_names):
        raise ValueError(
            f"chunk has {x.shape[0]} channels and buffer {buffer.samples.shape[0]}, but "
            f"{len(ch_names)} ch_names"
        )
    window, hop, left, right = _geometry(sfreq, streaming)

    samples = np.concatenate((buffer.samples, x), axis=1)
    end_index = buffer.first_index + samples.shape[1]
    last_start = end_index - window - right
    if last_start >= buffer.next_start:
        starts = np.arange(buffer.next_start, last_start + 1, hop, dtype=np.int64)
    else:
        starts = np.empty(0, dtype=np.int64)

    features = featurize_windows(
        samples,
        starts - buffer.first_index,
        sfreq,
        ch_names,
        streaming,
        onset_offset_s=buffer.origin_s,
        index_offset=buffer.first_index,
    )

    next_start = int(starts[-1]) + hop if starts.size else buffer.next_start
    keep_from = max(buffer.first_index, next_start - left)
    retained = np.array(samples[:, keep_from - buffer.first_index :], dtype=np.float64, copy=True)
    return (
        StreamBuffer(
            samples=retained,
            first_index=keep_from,
            next_start=next_start,
            origin_s=buffer.origin_s,
        ),
        features,
    )
