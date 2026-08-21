"""Band powers and feature naming."""

import pytest

pytestmark = pytest.mark.skip(reason="TODO(phase-1): implement after contract sign-off")


def test_band_power_lands_in_the_right_band() -> None:
    """A 10 Hz tone puts its power in alpha, not theta or beta."""


def test_band_power_is_resolution_invariant() -> None:
    """Trapezoid integration gives near-identical results at nperseg 160 and 320."""


def test_relative_powers_sum_to_one_per_channel() -> None:
    """The default bands tile 1-50 Hz, so relative values sum to 1.0."""


def test_relative_is_invariant_to_amplitude_scaling() -> None:
    """Scaling a channel by 10x leaves its relative band powers unchanged.

    This is the whole argument for relative being the headline: it is exactly the
    single-session amplitude confound that absolute power would encode.
    """


def test_absolute_log_is_not_invariant_to_amplitude_scaling() -> None:
    """The mirror of the above -- confirms the two modes genuinely differ."""


def test_feature_names_carry_the_normalization_prefix() -> None:
    """Single-mode output is still prefixed, so modes cannot be silently mixed."""


def test_feature_names_match_value_columns() -> None:
    """len(feature_names) equals values.shape[1] for every normalization mode."""


def test_feature_order_is_deterministic() -> None:
    """Same channels and bands produce the same column order every time."""


def test_empty_window_set_yields_empty_matrix() -> None:
    """Zero windows produce a (0, n_features) matrix, not an error."""
