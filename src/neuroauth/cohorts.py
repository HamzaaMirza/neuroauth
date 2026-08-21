"""Impostor-holdout selection.

CLAUDE.md hard rule 1: a held-out set of impostor subjects -- never enrolled, never
trained on -- must exist at all times. This module is what makes that set exist, and
it runs at ingest, before any Phase 1 result has been looked at.

Picking the holdout after seeing baseline results would be selection bias, and it
would compromise the Phase 2 EER before Phase 2 starts. The selection is therefore
deterministic in (seed, n_impostors) and the resulting list is recorded in
docs/DECISIONS.md.
"""

from collections.abc import Sequence

from neuroauth.config import HoldoutConfig


def select_impostor_holdout(
    subject_ids: Sequence[int],
    config: HoldoutConfig,
) -> frozenset[int]:
    """Choose the impostor-holdout subjects deterministically.

    Implemented as a seeded permutation of the sorted subject ids, truncated to
    config.n_impostors. Sorting first means the result does not depend on the order
    the caller happened to discover subjects in. Truncating a permutation means the
    selection is nested in n_impostors, so the count can be revised later without
    re-rolling subjects already committed to the holdout.

    Args:
        subject_ids: All available subject numbers. Duplicates are an error, not
            something to silently de-duplicate.
        config: Seed and holdout size.

    Returns:
        The impostor subject ids.

    Raises:
        ValueError: If subject_ids contains duplicates, or n_impostors is not in
            (0, len(subject_ids)).
    """
    raise NotImplementedError("TODO(phase-1): seeded nested holdout selection")


def partition_cohorts(
    subject_ids: Sequence[int],
    config: HoldoutConfig,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Split all subjects into the enrollable cohort and the impostor holdout.

    Phase 1's identification baseline trains and evaluates on the enrollable cohort
    only. The holdout subjects are not touched -- not for training, not for
    evaluation, not for a quick sanity check -- until Phase 2.

    Args:
        subject_ids: All available subject numbers.
        config: Seed and holdout size.

    Returns:
        (enrollable, impostor_holdout), both sorted ascending.
    """
    raise NotImplementedError("TODO(phase-1): cohort partition")


def assert_holdout_excluded(
    used_subject_ids: Sequence[int],
    impostor_holdout: frozenset[int],
) -> None:
    """Assert that no impostor-holdout subject appears in a training or eval set.

    Called on every training-set construction, not only in tests -- the same posture
    hard rule 5 takes for holdout integrity in Phase 3. This is the Phase 1
    ancestor of the assertions that will live in build_training_set.

    Args:
        used_subject_ids: Subjects appearing in the data about to be used.
        impostor_holdout: The reserved impostor set.

    Raises:
        AssertionError: If the two intersect, naming the leaked subjects.
    """
    raise NotImplementedError("TODO(phase-1): holdout leakage assertion")
