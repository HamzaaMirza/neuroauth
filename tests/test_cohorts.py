"""Impostor-holdout selection must be deterministic and nested."""

import pytest

pytestmark = pytest.mark.skip(reason="TODO(phase-1): implement after contract sign-off")


def test_selection_is_deterministic() -> None:
    """Same seed and count give the same holdout across processes."""


def test_selection_is_order_independent() -> None:
    """Shuffling the input subject list does not change the result."""


def test_selection_is_nested_in_count() -> None:
    """Raising n_impostors keeps every previously selected subject.

    This is what lets Phase 2 revise the count against a FAR stability argument
    without re-rolling a commitment already made.
    """


def test_cohorts_partition_exactly() -> None:
    """Enrollable and holdout are disjoint and together cover every subject."""


def test_duplicate_subject_ids_are_rejected() -> None:
    """Duplicates raise rather than being silently de-duplicated."""
