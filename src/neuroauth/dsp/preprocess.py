"""Filtering. Pure functions: numpy in, numpy out, no I/O, no MNE.

Everything here operates on (n_channels, n_samples) float64 in volts and returns a
new array of the same shape and dtype. Inputs are never mutated.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt

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
        ValueError: If the band is invalid, high_hz is at or above Nyquist, or order
            is below 1.
    """
    nyquist = sfreq / 2.0
    if not 0.0 < low_hz < high_hz < nyquist:
        raise ValueError(
            f"need 0 < low_hz < high_hz < Nyquist ({nyquist} Hz), got {low_hz}-{high_hz} Hz"
        )
    if order < 1:
        raise ValueError(f"order must be at least 1, got {order}")

    sos = butter(order, [low_hz, high_hz], btype="bandpass", fs=sfreq, output="sos")
    filtered: NDArray[np.float64] = sosfiltfilt(sos, np.asarray(data, dtype=np.float64), axis=-1)
    return filtered


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

    Raises:
        ValueError: If freq_hz or q is not positive, or n_harmonics is below 1.
    """
    if freq_hz <= 0.0 or q <= 0.0:
        raise ValueError(f"freq_hz and q must be positive, got {freq_hz} and {q}")
    if n_harmonics < 1:
        raise ValueError(f"n_harmonics must be at least 1, got {n_harmonics}")

    nyquist = sfreq / 2.0
    filtered: NDArray[np.float64] = np.array(data, dtype=np.float64, copy=True)
    for harmonic in range(1, n_harmonics + 1):
        target_hz = harmonic * freq_hz
        if target_hz >= nyquist:
            break
        b, a = iirnotch(target_hz, q, fs=sfreq)
        filtered = filtfilt(b, a, filtered, axis=-1)
    return filtered


def common_average_reference(data: NDArray[np.float64]) -> NDArray[np.float64]:
    """Subtract the across-channel mean at each sample.

    Args:
        data: (n_channels, n_samples), volts.

    Returns:
        (n_channels, n_samples) re-referenced copy. Rank drops by one; the
        channel-mean signal is no longer recoverable.
    """
    x = np.asarray(data, dtype=np.float64)
    referenced: NDArray[np.float64] = x - x.mean(axis=0, keepdims=True)
    return referenced


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

    Raises:
        ValueError: If data is not 2-D, or any filter setting is invalid.
    """
    x = np.asarray(data, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected (n_channels, n_samples), got shape {x.shape}")

    out = notch(x, sfreq, config.notch_freq, config.notch_q, config.notch_harmonics)
    out = bandpass(out, sfreq, config.bandpass_low, config.bandpass_high, config.bandpass_order)
    if config.common_average_reference:
        out = common_average_reference(out)
    return out
