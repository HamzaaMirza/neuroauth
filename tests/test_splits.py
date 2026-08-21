"""Leakage guards. These are the tests that decide whether any number means anything."""

import pytest

pytestmark = pytest.mark.skip(reason="TODO(phase-1): implement after contract sign-off")


def test_temporal_split_produces_no_overlapping_windows() -> None:
    """With a 2 s guard and 2 s windows, no train window shares samples with test."""


def test_guard_smaller_than_window_is_rejected() -> None:
    """guard_s below the window length raises rather than silently leaking."""


def test_assert_no_window_overlap_catches_a_deliberate_leak() -> None:
    """Hand-built overlapping indices must trip the assertion.

    A guard that has never been seen to fire is not a guard.
    """


def test_cross_condition_split_separates_runs() -> None:
    """Train indices are all eyes-open, test indices all eyes-closed."""


def test_cross_condition_split_requires_both_conditions_per_subject() -> None:
    """A subject missing one condition raises rather than skewing the label set."""


def test_concatenate_rejects_mismatched_feature_names() -> None:
    """Matrices built under different normalizations cannot be stacked."""


def test_holdout_subjects_never_appear_in_a_split() -> None:
    """assert_holdout_excluded fires if an impostor subject reaches training data."""
