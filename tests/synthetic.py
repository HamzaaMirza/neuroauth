"""Synthetic signal builders for the MNE-free pipeline tests.

Every signal here has a known spectral composition, so tests can assert that the
right band got the power rather than only that some number came out.
"""

import numpy as np
from numpy.typing import NDArray

from neuroauth.dsp.types import Condition, FeatureMatrix, QualityReport

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


def eeg_like_data(
    *, seed: int = 0, duration_s: float = DURATION_S, n_channels: int = 8
) -> NDArray[np.float64]:
    """(n_channels, n_samples) noise with a 1/f^2 power spectrum plus 10 Hz alpha, volts.

    The steep low-frequency spectrum puts most power in delta, as resting EEG does. That is
    where filter edge transients live, so this is the signal the edge-effect tests need.
    """
    rng = np.random.default_rng(seed)
    n_samples = round(duration_s * SFREQ)
    spectrum = np.fft.rfft(rng.normal(size=(n_channels, n_samples)), axis=-1)
    freqs = np.fft.rfftfreq(n_samples, 1.0 / SFREQ)
    amplitude = np.zeros_like(freqs)
    amplitude[1:] = 1.0 / freqs[1:]
    brown = np.fft.irfft(spectrum * amplitude, n=n_samples, axis=-1)
    brown *= 20e-6 / brown.std(axis=-1, keepdims=True)
    t = np.arange(n_samples) / SFREQ
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(n_channels, 1))
    data: NDArray[np.float64] = brown + ALPHA_V * np.sin(2.0 * np.pi * ALPHA_HZ * t + phases)
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


IDENTITY_BANDS = ("delta", "theta", "alpha", "beta", "gamma")
IDENTITY_FEATURE_NAMES = tuple(
    f"rel:C{channel:02d}:{band}" for channel in range(8) for band in IDENTITY_BANDS
)
_IDENTITY_BASIS = np.random.default_rng(99).normal(size=(6, len(IDENTITY_FEATURE_NAMES)))
_STATE_SHIFT = np.random.default_rng(98).normal(size=len(IDENTITY_FEATURE_NAMES))


def identity_feature_matrix(subject: int, run: int = 1, n_windows: int = 40) -> FeatureMatrix:
    """Relative band-power features with a learnable identity and a shared eyes-closed shift.

    log10 relative power = a subject-specific pattern in a 6-dimensional identity subspace,
    plus a shift common to every subject's eyes-closed run, plus window noise. An embedding
    fitted on some subjects should separate subjects it never saw.
    """
    n_features = len(IDENTITY_FEATURE_NAMES)
    identity = np.random.default_rng(subject).normal(size=6) @ _IDENTITY_BASIS
    noise = np.random.default_rng(1000 * subject + run).normal(size=(n_windows, n_features))
    log_values = -1.0 + 0.3 * identity + (0.8 if run == 2 else 0.0) * _STATE_SHIFT + 0.15 * noise
    return feature_matrix(
        subject_id=subject,
        run=run,
        condition="eyes_open" if run == 1 else "eyes_closed",
        n_windows=n_windows,
        feature_names=IDENTITY_FEATURE_NAMES,
        values=10.0**log_values,
    )


def feature_matrix(
    *,
    subject_id: int | None = 1,
    run: int | None = 1,
    condition: Condition | None = "eyes_open",
    n_windows: int = 60,
    hop_s: float = 1.0,
    feature_names: tuple[str, ...] = ("rel:C3:alpha", "rel:C4:alpha"),
    values: NDArray[np.float64] | None = None,
    window_ok: NDArray[np.bool_] | None = None,
    channel_ok: NDArray[np.bool_] | None = None,
    seed: int = 0,
) -> FeatureMatrix:
    """A FeatureMatrix for split and evaluation tests, clean unless told otherwise.

    Onsets start at 0 and advance by hop_s. Channels are read from the feature names
    (the "mode:channel:band" contract), in order of first appearance.
    """
    n_channels = len(dict.fromkeys(name.split(":")[1] for name in feature_names))
    if values is None:
        values = np.random.default_rng(seed).normal(size=(n_windows, len(feature_names)))
    if channel_ok is None:
        channel_ok = np.ones((n_windows, n_channels), dtype=np.bool_)
    if window_ok is None:
        window_ok = np.ones(n_windows, dtype=np.bool_)
    return FeatureMatrix(
        values=values,
        feature_names=feature_names,
        quality=QualityReport(
            window_ok=window_ok,
            channel_ok=channel_ok,
            flags=tuple(() for _ in range(n_windows)),
        ),
        onsets_s=np.arange(n_windows, dtype=np.float64) * hop_s,
        subject_id=subject_id,
        run=run,
        condition=condition,
    )
