"""Data structures carried through the signal pipeline.

All arrays are float64 in volts (MNE's native unit). Band powers are V^2/Hz unless
a normalization is applied. No structure defined here is ever persisted -- these are
in-memory inference-path objects (CLAUDE.md hard rule 3). The cancelable transform
in Phase 2 is what makes a vector storable.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

Condition = Literal["eyes_open", "eyes_closed", "task"]


@dataclass(frozen=True)
class Recording:
    """One continuous EEG recording, loaded and channel-name-standardized.

    Attributes:
        data: (n_channels, n_samples) float64, volts.
        sfreq: Sampling frequency in Hz.
        ch_names: Standardized 10-10 names, length n_channels. The trailing dots the
            eegmmidb EDF headers carry ("Fc5.", "C5..") have already been stripped by
            the loader.
        subject_id: PhysioNet subject number, 1-109.
        run: PhysioNet run number. 1 = eyes-open, 2 = eyes-closed, 3-14 = task.
        condition: Derived from run at load time, so downstream code never
            re-derives it from a magic number.
    """

    data: NDArray[np.float64]
    sfreq: float
    ch_names: tuple[str, ...]
    subject_id: int
    run: int
    condition: Condition


@dataclass(frozen=True)
class QualityReport:
    """Per-window and per-channel signal quality, computed alongside features.

    Never used to raise. The inference path returns a low-quality result and lets
    session logic decide (CLAUDE.md conventions).

    Attributes:
        window_ok: (n_windows,) bool. False when too many channels are bad.
        channel_ok: (n_windows, n_channels) bool.
        flags: Per-window tuple of human-readable reasons, e.g.
            ("flat_channels:3", "clipping:Fp1"). Empty tuple when clean.
    """

    window_ok: NDArray[np.bool_]
    channel_ok: NDArray[np.bool_]
    flags: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class WindowSet:
    """Fixed-length overlapping windows cut from one preprocessed recording.

    Attributes:
        data: (n_windows, n_channels, n_samples) float64, volts. Owns its memory --
            not a stride view -- so downstream detrending cannot corrupt neighbours.
        sfreq: Sampling frequency in Hz.
        ch_names: Length n_channels.
        onsets_s: (n_windows,) start time of each window, seconds from recording
            start. This is what makes leakage-free temporal splitting possible.
        subject_id: None for a live stream, where identity is claimed, not known.
        run: None for a live stream.
        condition: None for a live stream.
    """

    data: NDArray[np.float64]
    sfreq: float
    ch_names: tuple[str, ...]
    onsets_s: NDArray[np.float64]
    subject_id: int | None
    run: int | None
    condition: Condition | None


@dataclass(frozen=True)
class FeatureMatrix:
    """Band-power features for one WindowSet.

    Attributes:
        values: (n_windows, n_features) float64. Always finite -- never NaN or inf,
            even for windows flagged bad.
        feature_names: Length n_features, ordered to match columns. Column order is
            part of the model contract; it is persisted with the model, never with
            the data.
        quality: Aligned row-wise with `values`.
        onsets_s: (n_windows,) copied through from the WindowSet.
        subject_id: Copied through. None for a live stream.
        run: Copied through. None for a live stream.
        condition: Copied through. None for a live stream.
    """

    values: NDArray[np.float64]
    feature_names: tuple[str, ...]
    quality: QualityReport
    onsets_s: NDArray[np.float64]
    subject_id: int | None
    run: int | None
    condition: Condition | None
