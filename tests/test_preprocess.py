"""Filter behaviour. Synthetic signals only -- no dataset required."""

import numpy as np
import pytest

from neuroauth.config import PreprocessConfig
from neuroauth.dsp.preprocess import bandpass, common_average_reference, notch, preprocess
from neuroauth.dsp.types import Recording
from tests.synthetic import (
    ALPHA_HZ,
    ALPHA_V,
    MAINS_HZ,
    MAINS_V,
    SFREQ,
    amplitude_at,
    interior,
    rms,
    sine,
)


def _gain_db(filtered: np.ndarray, original: np.ndarray, margin_s: float = 2.0) -> float:
    return 20.0 * np.log10(rms(interior(filtered, margin_s)) / rms(interior(original, margin_s)))


def test_bandpass_passes_in_band_tone() -> None:
    """A 25 Hz tone survives the 1-50 Hz bandpass with amplitude largely intact."""
    x = sine(25.0)[None, :]
    assert _gain_db(bandpass(x, SFREQ, 1.0, 50.0, 4), x) == pytest.approx(0.0, abs=0.2)


@pytest.mark.parametrize("freq_hz", [0.2, 70.0])
def test_bandpass_rejects_out_of_band_tone(freq_hz: float) -> None:
    """Tones below the high-pass edge and above the low-pass edge lose at least 20 dB."""
    x = sine(freq_hz)[None, :]
    assert _gain_db(bandpass(x, SFREQ, 1.0, 50.0, 4), x, margin_s=10.0) <= -20.0


def test_bandpass_is_zero_phase() -> None:
    """filtfilt introduces no group delay: the output lines up with the input at lag 0."""
    x = interior(sine(ALPHA_HZ))
    y = interior(bandpass(sine(ALPHA_HZ)[None, :], SFREQ, 1.0, 50.0, 4)[0])
    lags = np.arange(-8, 9)  # half a 10 Hz period either side, so the peak is unambiguous
    correlation = [np.dot(x[8:-8], np.roll(y, lag)[8:-8]) for lag in lags]
    assert lags[int(np.argmax(correlation))] == 0


@pytest.mark.parametrize(("low_hz", "high_hz"), [(0.0, 50.0), (50.0, 1.0), (1.0, 80.0)])
def test_bandpass_rejects_invalid_band(low_hz: float, high_hz: float) -> None:
    with pytest.raises(ValueError):
        bandpass(sine(ALPHA_HZ)[None, :], SFREQ, low_hz, high_hz, 4)


def test_notch_removes_mains_tone() -> None:
    """A 60 Hz tone is attenuated far more than a 55 Hz neighbour."""
    mains = sine(60.0)[None, :]
    neighbour = sine(55.0)[None, :]
    assert _gain_db(notch(mains, SFREQ, 60.0, 30.0), mains) <= -26.0
    assert _gain_db(notch(neighbour, SFREQ, 60.0, 30.0), neighbour) >= -1.0


def test_notch_skips_harmonics_above_nyquist() -> None:
    """Requesting harmonics past Nyquist skips them instead of raising."""
    x = np.stack([sine(60.0), sine(ALPHA_HZ)])
    np.testing.assert_array_equal(
        notch(x, SFREQ, 60.0, 30.0, n_harmonics=3),
        notch(x, SFREQ, 60.0, 30.0, n_harmonics=1),
    )


def test_car_zeroes_channel_mean(synthetic_recording: Recording) -> None:
    """After CAR, the across-channel mean is zero at every sample."""
    referenced = common_average_reference(synthetic_recording.data)
    assert np.abs(synthetic_recording.data.mean(axis=0)).max() > 1e-7
    np.testing.assert_allclose(referenced.mean(axis=0), 0.0, atol=1e-18)


def test_preprocess_removes_mains_and_keeps_alpha(synthetic_recording: Recording) -> None:
    """The composed chain strips 60 Hz and leaves in-band alpha at full amplitude."""
    clean = interior(preprocess(synthetic_recording.data, SFREQ, PreprocessConfig()))
    assert amplitude_at(clean, MAINS_HZ).max() < 0.05 * MAINS_V
    np.testing.assert_allclose(amplitude_at(clean, ALPHA_HZ), ALPHA_V, rtol=0.05)


def test_preprocess_does_not_mutate_input(synthetic_recording: Recording) -> None:
    """The caller's array is unchanged after the full chain."""
    before = synthetic_recording.data.copy()
    preprocess(synthetic_recording.data, SFREQ, PreprocessConfig(common_average_reference=True))
    np.testing.assert_array_equal(synthetic_recording.data, before)


def test_preprocess_preserves_shape_and_dtype(synthetic_recording: Recording) -> None:
    """Output is (n_channels, n_samples) float64, same shape as input."""
    out = preprocess(synthetic_recording.data, SFREQ, PreprocessConfig())
    assert out.shape == synthetic_recording.data.shape
    assert out.dtype == np.float64


def test_preprocess_rejects_non_2d_input() -> None:
    with pytest.raises(ValueError):
        preprocess(sine(ALPHA_HZ), SFREQ, PreprocessConfig())
