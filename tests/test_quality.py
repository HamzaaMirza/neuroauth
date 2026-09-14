"""The quality mask never raises and never lets a NaN escape."""

import numpy as np
import pytest
from numpy.typing import NDArray

from neuroauth.config import FeatureConfig, QualityConfig
from neuroauth.dsp.features import assess_quality, extract_features
from neuroauth.dsp.types import WindowSet
from tests.synthetic import (
    CH_NAMES,
    CLIPPING_CHANNEL,
    FLAT_CHANNEL,
    NAN_CHANNEL,
    SFREQ,
    noise_windows,
)


def _window_set(data: NDArray[np.float64]) -> WindowSet:
    return WindowSet(
        data=data,
        sfreq=SFREQ,
        ch_names=CH_NAMES[: data.shape[1]],
        onsets_s=np.arange(data.shape[0], dtype=np.float64),
        subject_id=None,
        run=None,
        condition=None,
    )


@pytest.mark.parametrize(
    ("channel", "label"),
    [(FLAT_CHANNEL, "flat"), (CLIPPING_CHANNEL, "clipping"), (NAN_CHANNEL, "non_finite")],
)
def test_degraded_channel_is_flagged_by_name(
    degraded_windows: NDArray[np.float64], channel: int, label: str
) -> None:
    """Flat, clipping, and NaN channels are each marked bad and named in the flags."""
    report = assess_quality(degraded_windows, CH_NAMES, QualityConfig())
    assert not report.channel_ok[:, channel].any()
    assert all(f"{label}:{CH_NAMES[channel]}" in flags for flags in report.flags)


def test_only_degraded_channels_are_flagged(degraded_windows: NDArray[np.float64]) -> None:
    """Three bad channels of 64 is under the 20% threshold, so windows stay ok."""
    report = assess_quality(degraded_windows, CH_NAMES, QualityConfig())
    assert report.channel_ok[:, 3:].all()
    assert report.window_ok.all()


def test_clean_windows_have_no_flags() -> None:
    report = assess_quality(noise_windows(), CH_NAMES, QualityConfig())
    assert report.window_ok.all()
    assert report.channel_ok.all()
    assert all(flags == () for flags in report.flags)


@pytest.mark.parametrize(
    ("n_channels", "n_flat", "expected_ok"),
    [(64, 12, True), (64, 13, False), (25, 5, True), (25, 6, False)],
)
def test_window_marked_bad_above_bad_channel_fraction(
    n_channels: int, n_flat: int, expected_ok: bool
) -> None:
    """window_ok goes False once the bad fraction exceeds 0.2 -- exactly 0.2 is ok."""
    windows = noise_windows(n_windows=1, n_channels=n_channels)
    windows[:, :n_flat, :] = 0.0
    report = assess_quality(windows, CH_NAMES[:n_channels], QualityConfig())
    assert bool(report.window_ok[0]) is expected_ok
    has_fraction_flag = any(f.startswith("bad_channel_fraction:") for f in report.flags[0])
    assert has_fraction_flag is not expected_ok


def test_features_stay_finite_for_a_fully_degraded_window() -> None:
    """All-NaN and all-inf windows yield finite values plus window_ok False.

    The inference path must not raise on malformed input (CLAUDE.md conventions).
    """
    data = np.full((2, 64, 320), np.nan)
    data[1] = np.inf
    matrix = extract_features(
        _window_set(data), FeatureConfig(normalization="both"), QualityConfig()
    )
    assert matrix.values.shape == (2, 64 * 5 * 2)
    assert np.isfinite(matrix.values).all()
    assert not matrix.quality.window_ok.any()


def test_bad_channels_are_flagged_not_dropped(degraded_windows: NDArray[np.float64]) -> None:
    """Column count is fixed regardless of how many channels are bad."""
    matrix = extract_features(_window_set(degraded_windows), FeatureConfig(), QualityConfig())
    assert matrix.values.shape == (4, 64 * 5)
    assert np.isfinite(matrix.values).all()
    assert matrix.quality.channel_ok.shape == (4, 64)


def test_empty_windows_are_not_ok() -> None:
    report = assess_quality(np.zeros((2, 4, 0)), CH_NAMES[:4], QualityConfig())
    assert not report.window_ok.any()
    assert report.flags == (("empty_window",), ("empty_window",))


def test_structural_mismatch_raises() -> None:
    """Wrong ch_names length is a caller bug, not signal content, so it raises."""
    with pytest.raises(ValueError):
        assess_quality(noise_windows(), CH_NAMES[:10], QualityConfig())
