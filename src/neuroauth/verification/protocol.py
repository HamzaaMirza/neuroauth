"""The open-set evaluation protocol: who is unseen by what, and which data may choose thresholds.

Cohorts:

- **Impostor holdout H** (20, committed, D-008). Never fitted on, never enrolled, probes
  only. Used only for reporting: the headline FAR, FRR at FAR, and time-to-detect.
- **Enrollable cohort E** (89), dealt into N_FOLDS subject-disjoint folds. For fold k the
  embedding is fitted on E minus fold k, and fold-k subjects are enrolled and verified. So
  genuine users, not only impostors, are subjects the model has never seen. That is what
  deployment looks like: a new user enrolls without a retrain.
- **Cohort impostors.** Fold-k subjects act as impostors against each other, also unseen by
  that fold's model. Their scores are the only data allowed to choose an operating
  threshold. A threshold picked on H and then reported on H is selection on the test set.

Enrollment and probes (headline, cross-condition): enroll on the eyes-open recording and
probe with eyes-closed windows, for genuine and impostor comparisons alike. Taking impostor
probes from the same condition means a genuine comparison and an impostor comparison differ
only in whose EEG it is.

The key used for a protected comparison is always the claimed identity's. Transforming each
impostor probe with the impostor's own key is the "unstolen token" protocol error that
drives BioHash EERs toward zero in parts of the literature: the key does the discriminating,
not the biometric.
"""

from typing import Final

import numpy as np
from numpy.typing import NDArray

from neuroauth.config import StreamingConfig
from neuroauth.dsp.types import FeatureMatrix
from neuroauth.templates.enrollment import ProtectedTemplate
from neuroauth.verification.embedding import EmbeddingConfig
from neuroauth.verification.metrics import OperatingPoint, ScoreTable

N_FOLDS: Final = 5
FOLD_SEED: Final = 20260915
TEMPORAL_ENROLL_FRACTION: Final = 0.7
"""Temporal split only: enroll on the leading 70% of eyes-open, probe after a one-window
guard. Same value as Phase 1's TRAIN_FRACTION. All three fixed before any Phase 2 result."""

THRESHOLD_SELECTION_FAR: Final = 0.01
"""The deployed window-level threshold is the one achieving FAR <= 0.01 on cohort impostor
scores. It is reported on the holdout as VerificationReport.deployed."""


def subject_folds(
    subjects: tuple[int, ...], n_folds: int, seed: int
) -> tuple[tuple[int, ...], ...]:
    """Deal subjects into subject-disjoint folds.

    A seeded permutation of the sorted ids, dealt round-robin, sorted within each fold. The
    fold record is written into run_summary.json. It is not a committed file like the
    holdout, because folds are not a security boundary: every enrollable subject is
    evaluated in exactly one of them.

    Returns:
        n_folds tuples that partition subjects, with sizes differing by at most one.

    Raises:
        ValueError: On duplicate ids, or n_folds below 2 or above len(subjects).
    """
    raise NotImplementedError("TODO(phase-2): seeded round-robin folds")


def enrollment_and_probe_rows(
    matrices: list[FeatureMatrix],
    split_kind: str,
    *,
    window_s: float,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Row indices, into the concatenation of matrices, of enrollment and probe windows.

    cross_condition: enrollment = every eyes-open row, probes = every eyes-closed row.
    temporal: eyes-open only. Enrollment = the leading TEMPORAL_ENROLL_FRACTION, probes =
    the trailing block after a guard of window_s, via models.splits.temporal_split, which
    asserts no shared samples.

    For holdout subjects only the probe rows are ever used. score_fold enforces that.

    Raises:
        ValueError: On an unknown split_kind, or a subject missing a required recording.
        AssertionError: If any probe window shares samples with an enrollment window of the
            same recording.
    """
    raise NotImplementedError("TODO(phase-2): enrollment/probe selection")


def score_fold(
    matrices: list[FeatureMatrix],
    *,
    fold_index: int,
    fold_subjects: frozenset[int],
    train_subjects: frozenset[int],
    impostor_holdout: frozenset[int],
    split_kind: str,
    streaming: StreamingConfig,
    embedding_config: EmbeddingConfig,
    master_secret: bytes,
) -> tuple[ScoreTable, ScoreTable, tuple[int, ...]]:
    """Fit the fold's embedding, enroll the fold, and score every comparison in both domains.

    Each fold-k subject s is enrolled from its enrollment rows, then scored against its own
    probes (genuine), every other fold-k subject's probes (cohort impostors), and every
    holdout subject's probes (holdout impostors). Protected scores use s's key for every
    comparison against s.

    Enrollment embeddings and per-window embeddings are locals. The function returns score
    tables only.

    Args:
        matrices: process_recording_bounded output for every recording of E and H.
        fold_index: Recorded in ScoreTable.fold.
        fold_subjects: Enrolled and verified in this fold.
        train_subjects: The embedding is fitted on these.
        impostor_holdout: The committed holdout.
        split_kind: "cross_condition" or "temporal".
        streaming: Settings the matrices were built with.
        embedding_config: Hyperparameters.
        master_secret: An evaluation-only secret, never the deployment secret.

    Returns:
        (embedding_table, protected_table, failed_to_enroll), where failed_to_enroll lists
        fold subjects whose enrollment was refused. Those subjects contribute no genuine
        rows and are counted in the report.

    Raises:
        AssertionError: If train and fold intersect, train and H intersect, fold and H
            intersect, or a matrix's subject is in none of the three (hard rules 1 and 5).
    """
    raise NotImplementedError("TODO(phase-2): per-fold fit, enroll, score")


def select_operating_threshold(cohort_table: ScoreTable, target_far: float) -> OperatingPoint:
    """Choose a window-level threshold from cohort impostor scores only.

    Raises:
        AssertionError: If any row's source is a holdout subject. Thresholds never see H.
    """
    raise NotImplementedError("TODO(phase-2): threshold selection on cohort scores")


def random_pairing_control_eer(table: ScoreTable, *, seed: int) -> float:
    """Pooled EER with every genuine comparison replaced by a comparison with someone else.

    For each claimed subject s, the genuine scores are replaced by s's existing
    cohort-impostor scores against one other fold subject, chosen by a seeded derangement.
    Holdout impostor rows are left alone. Both sides are then impostor comparisons, so EER
    must sit near 0.5.

    What it catches: pairing and metric bugs that make "genuine" mean something other than
    same identity, such as a template built from the probe windows or misaligned subject
    columns. What it cannot catch: leakage into the embedding fit. That is ruled out
    structurally by the disjointness assertions in score_fold, as D-007 separates the
    shuffled-label control from assert_no_window_overlap.

    Raises:
        ValueError: If a fold has fewer than two enrolled subjects.
    """
    raise NotImplementedError("TODO(phase-2): derangement control")


def revocation_agreement(
    revoked: dict[int, ProtectedTemplate],
    reissued: dict[int, ProtectedTemplate],
    threshold: float,
) -> tuple[float, float]:
    """Bit agreement between each subject's revoked and reissued templates.

    Returns:
        (mean_agreement, fraction_accepted): the mean Hamming similarity over subjects, and
        the fraction of subjects whose revoked bits score >= threshold against their
        reissued template. Judged against metrics.REVOCATION_*.

    Raises:
        ValueError: If the subject sets differ, or any pair shares a key_version.
    """
    raise NotImplementedError("TODO(phase-2): revocation unlinkability")
