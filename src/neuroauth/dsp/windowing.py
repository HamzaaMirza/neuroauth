"""Fixed-length overlapping windows. Pure functions, no I/O."""

import numpy as np
from numpy.typing import NDArray

from neuroauth.config import WindowConfig
from neuroauth.dsp.types import Condition, Recording, WindowSet


def window_bounds(n_samples: int, window_samples: int, hop_samples: int) -> NDArray[np.int64]:
    """Compute window start indices.

    Split out from the framing itself so the index arithmetic -- the part that is
    easy to get wrong by one -- can be tested directly.

    Only windows that fit entirely inside the signal are emitted. A trailing
    remainder shorter than one window is dropped, never zero-padded: padding biases a
    window's PSD toward low frequencies.

    Args:
        n_samples: Length of the signal.
        window_samples: Window length in samples.
        hop_samples: Stride between window starts. Must be at least 1.

    Returns:
        (n_windows,) int64 start indices. Empty when the signal is shorter than one
        window.

    Raises:
        ValueError: If n_samples is negative, or window_samples or hop_samples is
            below 1.
    """
    if n_samples < 0:
        raise ValueError(f"n_samples must be non-negative, got {n_samples}")
    if window_samples < 1 or hop_samples < 1:
        raise ValueError(
            f"window_samples and hop_samples must be at least 1, "
            f"got {window_samples} and {hop_samples}"
        )
    if n_samples < window_samples:
        return np.empty(0, dtype=np.int64)
    return np.arange(0, n_samples - window_samples + 1, hop_samples, dtype=np.int64)


def _frames_at(
    data: NDArray[np.float64], starts: NDArray[np.int64], window_samples: int
) -> NDArray[np.float64]:
    offsets = np.arange(window_samples, dtype=np.int64)
    # Fancy indexing copies, so the result never aliases the source signal.
    frames = data[:, starts[:, None] + offsets[None, :]]
    return np.ascontiguousarray(frames.transpose(1, 0, 2))


def frame_signal(
    data: NDArray[np.float64],
    window_samples: int,
    hop_samples: int,
) -> NDArray[np.float64]:
    """Cut a multichannel signal into overlapping windows.

    Returns an owned, C-contiguous copy rather than a stride-trick view. At 50%
    overlap the copy costs 2x memory for one recording (~10 MB), and it means
    downstream in-place detrending cannot silently corrupt neighbouring windows.

    Args:
        data: (n_channels, n_samples), volts.
        window_samples: Window length in samples.
        hop_samples: Stride between window starts.

    Returns:
        (n_windows, n_channels, window_samples) float64.

    Raises:
        ValueError: If data is not 2-D, or the window geometry is invalid.
    """
    x = np.asarray(data, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected (n_channels, n_samples), got shape {x.shape}")
    starts = window_bounds(x.shape[1], window_samples, hop_samples)
    return _frames_at(x, starts, window_samples)


def _window_geometry(sfreq: float, config: WindowConfig) -> tuple[int, int]:
    if sfreq <= 0.0:
        raise ValueError(f"sfreq must be positive, got {sfreq}")
    if not 0.0 <= config.overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {config.overlap}")
    window_samples = round(config.window_s * sfreq)
    if window_samples < 1:
        raise ValueError(f"window of {config.window_s} s at {sfreq} Hz is under one sample")
    hop_samples = round(window_samples * (1.0 - config.overlap))
    if hop_samples < 1:
        raise ValueError(
            f"overlap {config.overlap} on a {window_samples}-sample window gives a hop of 0"
        )
    return window_samples, hop_samples


def _build_window_set(
    data: NDArray[np.float64],
    sfreq: float,
    ch_names: tuple[str, ...],
    config: WindowConfig,
    *,
    onset_offset_s: float,
    subject_id: int | None,
    run: int | None,
    condition: Condition | None,
) -> WindowSet:
    window_samples, hop_samples = _window_geometry(sfreq, config)
    x = np.asarray(data, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected (n_channels, n_samples), got shape {x.shape}")
    if x.shape[0] != len(ch_names):
        raise ValueError(f"data has {x.shape[0]} channels but {len(ch_names)} ch_names")

    starts = window_bounds(x.shape[1], window_samples, hop_samples)
    return WindowSet(
        data=_frames_at(x, starts, window_samples),
        sfreq=sfreq,
        ch_names=ch_names,
        onsets_s=starts.astype(np.float64) / sfreq + onset_offset_s,
        subject_id=subject_id,
        run=run,
        condition=condition,
    )


def window_recording(recording: Recording, config: WindowConfig) -> WindowSet:
    """Window a preprocessed recording, carrying provenance through.

    Args:
        recording: A Recording whose `data` has already been preprocessed. Windowing
            does not filter -- the caller controls the order.
        config: Window length and overlap.

    Returns:
        A WindowSet with subject_id, run, and condition copied from the recording,
        and onsets_s in seconds from recording start.

    Raises:
        ValueError: If overlap is outside [0, 1) or the resulting hop is 0.
    """
    return _build_window_set(
        recording.data,
        recording.sfreq,
        recording.ch_names,
        config,
        onset_offset_s=0.0,
        subject_id=recording.subject_id,
        run=recording.run,
        condition=recording.condition,
    )


def window_stream_chunk(
    buffer: NDArray[np.float64],
    sfreq: float,
    ch_names: tuple[str, ...],
    config: WindowConfig,
    *,
    onset_offset_s: float = 0.0,
) -> WindowSet:
    """Window an in-memory stream buffer with no known subject identity.

    Same framing as window_recording, but subject_id, run, and condition are None,
    because a live stream carries a *claimed* identity rather than a known one.
    Present in Phase 1 so the Phase 2 streaming path reuses this code instead of
    growing a parallel implementation that drifts from it.

    A buffer shorter than one window yields an empty WindowSet, not an error.

    Args:
        buffer: (n_channels, n_samples), volts, already preprocessed.
        sfreq: Hz.
        ch_names: Length n_channels.
        config: Window length and overlap.
        onset_offset_s: Seconds elapsed before this buffer, so onsets_s stays
            monotonic across successive chunks.

    Returns:
        A WindowSet with subject_id, run, and condition set to None.

    Raises:
        ValueError: If the buffer is not 2-D, its channel axis does not match
            ch_names, or the window config is invalid. These are structural caller
            errors; the Phase 2 WebSocket handler validates frames before calling.
    """
    return _build_window_set(
        buffer,
        sfreq,
        ch_names,
        config,
        onset_offset_s=onset_offset_s,
        subject_id=None,
        run=None,
        condition=None,
    )
