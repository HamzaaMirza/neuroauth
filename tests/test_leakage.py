"""The deliberately leaky split, and counting the windows it leaks (D-017)."""

import numpy as np
import pytest

from neuroauth.dsp.types import FeatureMatrix
from neuroauth.models.splits import (
    assert_no_window_overlap,
    count_overlapping_test_windows,
    leaky_random_split,
    temporal_split,
)
from tests.synthetic import feature_matrix

WINDOW_S = 2.0
N_ROWS = 6 * 60


def _cohort() -> list[FeatureMatrix]:
    return [
        feature_matrix(subject_id=subject, run=run, condition=condition, seed=10 * subject + run)
        for subject in (1, 2, 3)
        for run, condition in ((1, "eyes_open"), (2, "eyes_closed"))
    ]


def test_leaky_split_is_a_disjoint_partition() -> None:
    train, test = leaky_random_split(_cohort(), 0.7, seed=1)
    assert np.intersect1d(train, test).size == 0
    np.testing.assert_array_equal(np.union1d(train, test), np.arange(N_ROWS))
    assert train.size == round(0.7 * N_ROWS)


def test_leaky_split_is_seeded() -> None:
    first = leaky_random_split(_cohort(), 0.7, seed=1)
    np.testing.assert_array_equal(first[0], leaky_random_split(_cohort(), 0.7, seed=1)[0])
    assert not np.array_equal(first[0], leaky_random_split(_cohort(), 0.7, seed=2)[0])


def test_leaky_split_really_leaks() -> None:
    """Its output fails the guard every real split passes, and most test windows leak.

    With 1 s hops and 2 s windows each window overlaps its two neighbours, so about
    1 - 0.3^2 = 91% of test windows share samples with a training window.
    """
    matrices = _cohort()
    train, test = leaky_random_split(matrices, 0.7, seed=1)
    with pytest.raises(AssertionError, match="shares samples"):
        assert_no_window_overlap(matrices, train, test, WINDOW_S)
    fraction = count_overlapping_test_windows(matrices, train, test, WINDOW_S) / test.size
    assert 0.8 < fraction < 0.98


def test_guarded_temporal_split_leaks_nothing() -> None:
    matrices = _cohort()
    train, test = temporal_split(matrices, 0.7, window_s=WINDOW_S, guard_s=WINDOW_S)
    assert count_overlapping_test_windows(matrices, train, test, WINDOW_S) == 0


def test_count_matches_a_hand_built_boundary() -> None:
    """Test onset 40 s is 1 s from train onset 39 s; 41 s is exactly one window away."""
    count = count_overlapping_test_windows(
        [feature_matrix()], np.arange(0, 40), np.arange(40, 60), WINDOW_S
    )
    assert count == 1


@pytest.mark.parametrize("train_fraction", [0.0, 1.0])
def test_leaky_split_rejects_degenerate_fraction(train_fraction: float) -> None:
    with pytest.raises(ValueError):
        leaky_random_split(_cohort(), train_fraction, seed=1)
