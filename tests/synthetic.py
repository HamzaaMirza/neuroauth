"""Synthetic signal builders for the MNE-free pipeline tests.

Every signal here has a known spectral composition, so tests can assert that the
right band got the power rather than only that some number came out.
"""

import numpy as np
from numpy.typing import NDArray

SFREQ = 160.0
N_CHANNELS = 64
DURATION_S = 60.0
CH_NAMES: tuple[str, ...] = tuple(f"Ch{i:02d}" for i in range(1, N_CHANNELS + 1))

FLAT_CHANNEL = 0
CLIPPING_CHANNEL = 1
NAN_CHANNEL = 2

ALPHA_HZ, ALPHA_V = 10.0, 20e-6
BETA_HZ, BETA_V = 20.0, 5e-6
MAINS_HZ, MAINS_V = 60.0, 10e-6


def sine(
    freq_hz: float,
    *,
    duration_s: float = DURATION_S,
    amplitude_v: float = 10e-6,
    phase: float = 0.0,
) -> NDArray[np.float64]:
    """(n_samples,) sinusoid at SFREQ, volts."""
    t = np.arange(round(duration_s * SFREQ)) / SFREQ
    return amplitude_v * np.sin(2.0 * np.pi * freq_hz * t + phase)


def rms(x: NDArray[np.float64]) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def interior(x: NDArray[np.float64], margin_s: float = 2.0) -> NDArray[np.float64]:
    """Drop margin_s from both ends of the last axis, excluding filter edge transients."""
    margin = round(margin_s * SFREQ)
    return x[..., margin:-margin]


def amplitude_at(x: NDArray[np.float64], freq_hz: float) -> NDArray[np.float64]:
    """Amplitude of the freq_hz component along the last axis, by projection."""
    t = np.arange(x.shape[-1]) / SFREQ
    projection = x @ np.exp(-2j * np.pi * freq_hz * t)
    amplitude: NDArray[np.float64] = 2.0 * np.abs(projection) / x.shape[-1]
    return amplitude


def recording_data(*, seed: int = 0, duration_s: float = DURATION_S) -> NDArray[np.float64]:
    """(N_CHANNELS, n_samples) of alpha, beta, and mains tones over a noise floor, volts.

    Each channel gets independent random phases, so the across-channel mean is not
    trivially zero before common average referencing.
    """
    rng = np.random.default_rng(seed)
    n_samples = round(duration_s * SFREQ)
    t = np.arange(n_samples) / SFREQ
    data = rng.normal(0.0, 2e-6, size=(N_CHANNELS, n_samples))
    for freq_hz, amplitude_v in ((ALPHA_HZ, ALPHA_V), (BETA_HZ, BETA_V), (MAINS_HZ, MAINS_V)):
        phases = rng.uniform(0.0, 2.0 * np.pi, size=(N_CHANNELS, 1))
        data += amplitude_v * np.sin(2.0 * np.pi * freq_hz * t + phases)
    return data


def noise_windows(
    *, seed: int = 0, n_windows: int = 4, n_channels: int = N_CHANNELS, n_samples: int = 320
) -> NDArray[np.float64]:
    """(n_windows, n_channels, n_samples) Gaussian noise at 10 uV std -- clean by
    every quality threshold."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 10e-6, size=(n_windows, n_channels, n_samples))


def degraded_windows(*, seed: int = 0) -> NDArray[np.float64]:
    """Clean noise windows with one flat, one clipping, and one NaN channel."""
    windows = noise_windows(seed=seed)
    windows[:, FLAT_CHANNEL, :] = 5e-6
    windows[:, CLIPPING_CHANNEL, ::2] = 1e-3
    windows[:, CLIPPING_CHANNEL, 1::2] = -1e-3
    windows[:, NAN_CHANNEL, 10] = np.nan
    return windows
