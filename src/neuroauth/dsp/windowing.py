"""Fixed-length overlapping windows. Pure functions, no I/O."""

import numpy as np
from numpy.typing import NDArray

from neuroauth.config import WindowConfig
from neuroauth.dsp.types import Recording, WindowSet


def window_bounds(
    n_samples: int,
    window_samples: int,
    hop_samples: int,
    drop_partial: bool,
) -> NDArray[np.int64]:
    """Compute window start indices.

    Split out from the framing itself so the index arithmetic -- the part that is
    easy to get wrong by one -- can be tested directly.

    Args:
        n_samples: Length of the signal.
        window_samples: Window length in samples.
        hop_samples: Stride between window starts. Must be at least 1.
        drop_partial: Exclude a final window that would run past the end. When
            False, no window is emitted past the end either: a partial window is
            never zero-padded, since padding biases its PSD toward low frequencies.
            The flag exists to make that explicit rather than implicit.

    Returns:
        (n_windows,) int64 start indices. Empty when the signal is shorter than one
        window.
    """
    raise NotImplementedError("TODO(phase-1): window start indices")


def frame_signal(
    data: NDArray[np.float64],
    window_samples: int,
    hop_samples: int,
    drop_partial: bool = True,
) -> NDArray[np.float64]:
    """Cut a multichannel signal into overlapping windows.

    Returns an owned, C-contiguous copy rather than a stride-trick view. At 50%
    overlap the copy costs 2x memory for one recording (~10 MB), and it means
    downstream in-place detrending cannot silently corrupt neighbouring windows.

    Args:
        data: (n_channels, n_samples), volts.
        window_samples: Window length in samples.
        hop_samples: Stride between window starts.
        drop_partial: See window_bounds.

    Returns:
        (n_windows, n_channels, window_samples) float64.
    """
    raise NotImplementedError("TODO(phase-1): frame into owned windows")


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
    raise NotImplementedError("TODO(phase-1): window a recording")


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

    Args:
        buffer: (n_channels, n_samples), volts, already preprocessed.
        sfreq: Hz.
        ch_names: Length n_channels.
        config: Window length and overlap.
        onset_offset_s: Seconds elapsed before this buffer, so onsets_s stays
            monotonic across successive chunks.

    Returns:
        A WindowSet with subject_id, run, and condition set to None.
    """
    raise NotImplementedError("TODO(phase-1): window a stream buffer")
