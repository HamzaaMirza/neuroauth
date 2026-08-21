"""Filter behaviour. Synthetic signals only -- no dataset required."""

import pytest

pytestmark = pytest.mark.skip(reason="TODO(phase-1): implement after contract sign-off")


def test_bandpass_passes_in_band_tone() -> None:
    """A 25 Hz tone survives the 1-50 Hz bandpass with amplitude largely intact."""


def test_bandpass_rejects_out_of_band_tone() -> None:
    """A 70 Hz tone is attenuated by at least 20 dB."""


def test_bandpass_is_zero_phase() -> None:
    """filtfilt introduces no group delay: peak positions are unmoved."""


def test_notch_removes_mains_tone() -> None:
    """A 60 Hz tone is attenuated far more than a 55 Hz neighbour."""


def test_notch_skips_harmonics_above_nyquist() -> None:
    """Requesting harmonics past Nyquist skips them instead of raising."""


def test_car_zeroes_channel_mean() -> None:
    """After CAR, the across-channel mean is zero at every sample."""


def test_preprocess_does_not_mutate_input() -> None:
    """The caller's array is unchanged after the full chain."""


def test_preprocess_preserves_shape_and_dtype() -> None:
    """Output is (n_channels, n_samples) float64, same shape as input."""
