"""Filtering. Pure functions: numpy in, numpy out, no I/O, no MNE.

Everything here operates on (n_channels, n_samples) float64 in volts and returns a
new array of the same shape and dtype. Inputs are never mutated.
"""

import numpy as np
from numpy.typing import NDArray

from neuroauth.config import PreprocessConfig


def bandpass(
    data: NDArray[np.float64],
    sfreq: float,
    low_hz: float,
    high_hz: float,
    order: int,
) -> NDArray[np.float64]:
    """Zero-phase Butterworth bandpass, applied per channel.

    Uses second-order sections with sosfiltfilt. Transfer-function form is
    numerically unstable at this order-to-sample-rate ratio, and filtfilt keeps the
    phase response flat so band powers are not time-shifted relative to each other
    across frequencies.

    Args:
        data: (n_channels, n_samples), volts.
        sfreq: Hz.
        low_hz: High-pass edge.
        high_hz: Low-pass edge. Must be below Nyquist.
        order: Butterworth order per direction.

    Returns:
        (n_channels, n_samples) filtered copy.

    Raises:
        ValueError: If the band is invalid or high_hz is at or above Nyquist.
    """
    raise NotImplementedError("TODO(phase-1): sos butterworth bandpass")


def notch(
    data: NDArray[np.float64],
    sfreq: float,
    freq_hz: float,
    q: float,
    n_harmonics: int = 1,
) -> NDArray[np.float64]:
    """Zero-phase IIR notch at the mains frequency and optional harmonics.

    Args:
        data: (n_channels, n_samples), volts.
        sfreq: Hz.
        freq_hz: Fundamental to remove. 60.0 for eegmmidb.
        q: Notch quality factor. Higher = narrower stopband.
        n_harmonics: Harmonics to remove, including the fundamental. Harmonics at or
            above Nyquist are skipped rather than raising.

    Returns:
        (n_channels, n_samples) filtered copy.
    """
    raise NotImplementedError("TODO(phase-1): iirnotch + filtfilt")


def common_average_reference(data: NDArray[np.float64]) -> NDArray[np.float64]:
    """Subtract the across-channel mean at each sample.

    Args:
        data: (n_channels, n_samples), volts.

    Returns:
        (n_channels, n_samples) re-referenced copy. Rank drops by one; the
        channel-mean signal is no longer recoverable.
    """
    raise NotImplementedError("TODO(phase-1): common average reference")


def preprocess(
    data: NDArray[np.float64],
    sfreq: float,
    config: PreprocessConfig,
) -> NDArray[np.float64]:
    """Apply the full chain: notch, then bandpass, then optional CAR.

    Notch runs before the bandpass so the 60 Hz peak is removed while it is still
    well inside the signal band, rather than relying on the 50 Hz passband edge to
    attenuate it.

    Args:
        data: (n_channels, n_samples), volts. Not mutated.
        sfreq: Hz.
        config: Filter settings.

    Returns:
        (n_channels, n_samples) preprocessed copy.
    """
    raise NotImplementedError("TODO(phase-1): compose notch -> bandpass -> car")
