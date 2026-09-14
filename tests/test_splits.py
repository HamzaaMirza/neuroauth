"""Leakage guards. These are the tests that decide whether any number means anything."""

import numpy as np
import pytest

from neuroauth.cohorts import assert_holdout_excluded
from neuroauth.dsp.types import FeatureMatrix
from neuroauth.models.splits import (
    assert_no_window_overlap,
    concatenate,
    cross_condition_split,
    temporal_split,
)
from tests.synthetic import feature_matrix

WINDOW_S = 2.0
N_WINDOWS = 60


def _baseline_pair(subject_id: int) -> list[FeatureMatrix]:
    return [
        feature_matrix(subject_id=subject_id, run=1, condition="eyes_open", seed=subject_id),
        feature_matrix(
            subject_id=subject_id, run=2, condition="eyes_closed", seed=100 + subject_id
        ),
    ]


def _cohort(n_subjects: int = 3) -> list[FeatureMatrix]:
    return [matrix for subject in range(1, n_subjects + 1) for matrix in _baseline_pair(subject)]


def test_temporal_split_produces_no_overlapping_windows() -> None:
    """With a 2 s guard and 2 s windows, no train window shares samples with test."""
    matrices = _cohort()
    train, test = temporal_split(matrices, 0.7, window_s=WINDOW_S, guard_s=WINDOW_S)
    assert np.intersect1d(train, test).size == 0

    onsets = np.concatenate([matrix.onsets_s for matrix in matrices])
    recording = np.repeat(np.arange(len(matrices)), N_WINDOWS)
    for k in range(len(matrices)):
        last_train = onsets[train][recording[train] == k].max()
        first_test = onsets[test][recording[test] == k].min()
        assert first_test - last_train >= WINDOW_S


def test_temporal_split_takes_leading_train_and_trailing_test() -> None:
    """61 s timeline, 70%: train onsets 0-42, test onsets 45-59, 43-44 discarded."""
    train, test = temporal_split(_cohort(1), 0.7, window_s=WINDOW_S, guard_s=WINDOW_S)
    np.testing.assert_array_equal(train, np.r_[0:43, 60:103])
    np.testing.assert_array_equal(test, np.r_[45:60, 105:120])


@pytest.mark.parametrize("guard_s", [0.0, 1.0, 1.999])
def test_guard_smaller_than_window_is_rejected(guard_s: float) -> None:
    """guard_s below the window length raises rather than silently leaking."""
    with pytest.raises(ValueError, match="guard_s"):
        temporal_split(_cohort(), 0.7, window_s=WINDOW_S, guard_s=guard_s)


@pytest.mark.parametrize("train_fraction", [0.0, 1.0])
def test_train_fraction_must_be_strictly_between_zero_and_one(train_fraction: float) -> None:
    with pytest.raises(ValueError):
        temporal_split(_cohort(), train_fraction, window_s=WINDOW_S, guard_s=WINDOW_S)


def test_recording_left_without_test_windows_is_rejected() -> None:
    """A subject must never silently vanish from one side of the evaluation."""
    with pytest.raises(ValueError, match="contributes"):
        temporal_split([feature_matrix(n_windows=3)], 0.7, window_s=WINDOW_S, guard_s=WINDOW_S)


def test_assert_no_window_overlap_catches_a_deliberate_leak() -> None:
    """Hand-built overlapping indices must trip the assertion.

    A guard that has never been seen to fire is not a guard.
    """
    with pytest.raises(AssertionError, match="shares samples"):
        assert_no_window_overlap(_cohort(1), np.arange(0, 40), np.arange(40, 60), WINDOW_S)


def test_windows_exactly_one_window_apart_do_not_overlap() -> None:
    """Onsets 39 s and 41 s with 2 s windows touch but share no sample."""
    assert_no_window_overlap(_cohort(1), np.arange(0, 40), np.arange(41, 60), WINDOW_S)


def test_index_in_both_train_and_test_is_caught() -> None:
    with pytest.raises(AssertionError, match="both train and test"):
        assert_no_window_overlap(_cohort(1), np.arange(0, 30), np.arange(29, 60), WINDOW_S)


def test_out_of_range_index_is_caught() -> None:
    with pytest.raises(AssertionError, match="outside"):
        assert_no_window_overlap(_cohort(1), np.arange(0, 10), np.array([999]), WINDOW_S)


def test_same_onsets_in_different_recordings_are_not_overlap() -> None:
    matrices = _cohort(1)
    assert_no_window_overlap(matrices, np.arange(0, 60), np.arange(60, 120), WINDOW_S)


def test_cross_condition_split_separates_runs() -> None:
    """Train indices are all eyes-open, test indices all eyes-closed."""
    matrices = _cohort()
    train, test = cross_condition_split(matrices, "eyes_open", "eyes_closed")
    conditions = np.repeat([matrix.condition for matrix in matrices], N_WINDOWS)
    assert set(conditions[train]) == {"eyes_open"}
    assert set(conditions[test]) == {"eyes_closed"}
    assert train.size + test.size == len(matrices) * N_WINDOWS


def test_cross_condition_split_ignores_task_runs() -> None:
    matrices = [*_cohort(2), feature_matrix(subject_id=1, run=4, condition="task", seed=9)]
    train, test = cross_condition_split(matrices, "eyes_open", "eyes_closed")
    assert train.size + test.size == 4 * N_WINDOWS


def test_cross_condition_split_requires_both_conditions_per_subject() -> None:
    """A subject missing one condition raises rather than skewing the label set."""
    matrices = [*_cohort(2), feature_matrix(subject_id=3, run=1, condition="eyes_open")]
    with pytest.raises(ValueError, match=r"\[3\]"):
        cross_condition_split(matrices, "eyes_open", "eyes_closed")


def test_cross_condition_split_rejects_same_condition() -> None:
    with pytest.raises(ValueError):
        cross_condition_split(_cohort(), "eyes_open", "eyes_open")


def test_cross_condition_split_rejects_absent_condition() -> None:
    eyes_open_only = [matrix for matrix in _cohort() if matrix.condition == "eyes_open"]
    with pytest.raises(ValueError, match="eyes_closed"):
        cross_condition_split(eyes_open_only, "eyes_open", "eyes_closed")


def test_concatenate_labels_rows_by_subject() -> None:
    x, y, names = concatenate(_cohort(2))
    assert x.shape == (4 * N_WINDOWS, 2)
    np.testing.assert_array_equal(y, np.repeat([1, 1, 2, 2], N_WINDOWS))
    assert names == ("rel:C3:alpha", "rel:C4:alpha")


def test_concatenate_rejects_mismatched_feature_names() -> None:
    """Matrices built under different normalizations cannot be stacked."""
    relative = feature_matrix(feature_names=("rel:C3:alpha",))
    absolute = feature_matrix(subject_id=2, feature_names=("abs:C3:alpha",))
    with pytest.raises(ValueError, match="feature_names"):
        concatenate([relative, absolute])


def test_concatenate_rejects_unlabelled_matrix() -> None:
    with pytest.raises(ValueError, match="subject_id"):
        concatenate([feature_matrix(subject_id=None)])


def test_holdout_subjects_never_appear_in_a_split() -> None:
    """assert_holdout_excluded fires if an impostor subject reaches training data."""
    matrices = _cohort(3)
    _, y, _ = concatenate(matrices)
    train, test = cross_condition_split(matrices, "eyes_open", "eyes_closed")
    assert_holdout_excluded(y[train], frozenset({7, 8}))
    with pytest.raises(AssertionError, match=r"\[2\]"):
        assert_holdout_excluded(y[test], frozenset({2, 9}))
