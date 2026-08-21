"""The quality mask never raises and never lets a NaN escape."""

import pytest

pytestmark = pytest.mark.skip(reason="TODO(phase-1): implement after contract sign-off")


def test_flat_channel_is_flagged() -> None:
    """A constant channel trips flat_std_v."""


def test_clipping_channel_is_flagged() -> None:
    """A channel exceeding max_peak_to_peak_v trips the clipping check."""


def test_nan_channel_is_flagged() -> None:
    """Non-finite samples are caught before they reach the PSD."""


def test_window_marked_bad_above_bad_channel_fraction() -> None:
    """window_ok goes False once too many channels are bad."""


def test_features_stay_finite_for_a_fully_degraded_window() -> None:
    """An all-NaN window yields finite values plus window_ok False -- never a raise.

    The inference path must not raise on malformed input (CLAUDE.md conventions).
    """


def test_bad_channels_are_flagged_not_dropped() -> None:
    """Column count is fixed regardless of how many channels are bad."""
