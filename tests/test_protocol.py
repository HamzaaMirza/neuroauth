"""The open-set protocol: folds, cohort assertions, the claimed identity's key, thresholds that
never see the holdout, the pairing control, cross-fitting, and revocation."""

from datetime import UTC, datetime

import numpy as np
import pytest

from neuroauth.config import StreamingConfig
from neuroauth.dsp.types import FeatureMatrix
from neuroauth.models.splits import assert_no_window_overlap
from neuroauth.templates.cancelable import (
    derive_key,
    hamming_similarity,
    projection_matrix,
    protect,
)
from neuroauth.templates.enrollment import subject_ref
from neuroauth.verification.decision import decision_llr, fit_decision_layer
from neuroauth.verification.embedding import EmbeddingConfig, embed
from neuroauth.verification.metrics import (
    check_score_table,
    concatenate_tables,
    eer_value,
    equal_error_rate,
    error_rates,
    select_rows,
    split_scores,
)
from neuroauth.verification.protocol import (
    FoldScores,
    cross_fit_decision_table,
    enrollment_and_probe_rows,
    random_pairing_control_eer,
    reissue_fold_templates,
    revocation_agreement,
    score_fold,
    select_operating_threshold,
    subject_folds,
)
from tests.synthetic import identity_feature_matrix

SECRET = bytes(range(32))
NOW = datetime(2026, 9, 15, tzinfo=UTC)
STREAMING = StreamingConfig()
ENROLLABLE = tuple(range(1, 31))
HOLDOUT = frozenset({90, 91, 92, 93})
FOLDS = subject_folds(ENROLLABLE, 3, seed=7)
SHORT_ENROLLMENT_SUBJECT = FOLDS[0][0]


def build_matrices() -> list[FeatureMatrix]:
    matrices = []
    for subject in (*ENROLLABLE, *sorted(HOLDOUT)):
        for run in (1, 2):
            n_windows = 10 if (subject, run) == (SHORT_ENROLLMENT_SUBJECT, 1) else 40
            matrices.append(identity_feature_matrix(subject, run, n_windows=n_windows))
    return matrices


MATRICES = build_matrices()


def run_fold(index: int, split_kind: str = "cross_condition") -> FoldScores:
    fold = frozenset(FOLDS[index])
    return score_fold(
        MATRICES,
        fold_index=index,
        fold_subjects=fold,
        train_subjects=frozenset(ENROLLABLE) - fold,
        impostor_holdout=HOLDOUT,
        split_kind=split_kind,
        streaming=STREAMING,
        embedding_config=EmbeddingConfig(),
        master_secret=SECRET,
        enrolled_at=NOW,
    )


@pytest.fixture(scope="module")
def folds() -> list[FoldScores]:
    return [run_fold(k) for k in range(len(FOLDS))]


def test_subject_folds_partition_balance_and_determinism() -> None:
    assert sorted(s for fold in FOLDS for s in fold) == list(ENROLLABLE)
    assert {len(fold) for fold in FOLDS} == {10}
    assert subject_folds(ENROLLABLE, 3, seed=7) == FOLDS
    assert subject_folds(ENROLLABLE, 3, seed=8) != FOLDS
    with pytest.raises(ValueError, match="duplicate"):
        subject_folds((1, 1, 2), 2, seed=0)
    with pytest.raises(ValueError, match="n_folds"):
        subject_folds((1, 2), 3, seed=0)


def test_cross_condition_enrolls_eyes_open_and_probes_eyes_closed() -> None:
    matrices = [identity_feature_matrix(s, run) for s in (1, 2) for run in (1, 2)]
    enroll, probe = enrollment_and_probe_rows(matrices, "cross_condition", window_s=2.0)
    np.testing.assert_array_equal(enroll, np.r_[0:40, 80:120])
    np.testing.assert_array_equal(probe, np.r_[40:80, 120:160])


def test_temporal_split_probes_never_share_samples_with_enrollment() -> None:
    matrices = [identity_feature_matrix(s, run) for s in (1, 2) for run in (1, 2)]
    enroll, probe = enrollment_and_probe_rows(matrices, "temporal", window_s=2.0)
    assert np.intersect1d(enroll, probe).size == 0
    assert all(row < 40 or 80 <= row < 120 for row in (*enroll, *probe))  # eyes-open only
    assert_no_window_overlap(matrices, enroll, probe, window_s=2.0)


@pytest.mark.parametrize(
    ("fold", "train", "holdout", "match"),
    [
        (frozenset({1, 2}), frozenset({2, 3}), HOLDOUT, "training and fold"),
        (frozenset({1}), frozenset({2, 90}), HOLDOUT, "training and holdout"),
        (frozenset({1, 90}), frozenset({2}), HOLDOUT, "fold and holdout"),
        (frozenset({1}), frozenset({2, 3}), HOLDOUT, "no cohort"),
    ],
)
def test_score_fold_asserts_cohort_boundaries(
    fold: frozenset[int], train: frozenset[int], holdout: frozenset[int], match: str
) -> None:
    with pytest.raises(AssertionError, match=match):
        score_fold(
            MATRICES,
            fold_index=0,
            fold_subjects=fold,
            train_subjects=train,
            impostor_holdout=holdout,
            split_kind="cross_condition",
            streaming=STREAMING,
            embedding_config=EmbeddingConfig(),
            master_secret=SECRET,
            enrolled_at=NOW,
        )


def test_a_fold_scores_every_comparison_it_should(folds: list[FoldScores]) -> None:
    fold = folds[0]
    for table in (fold.embedding_table, fold.protected_table):
        check_score_table(table, HOLDOUT)
    n_enrolled = len(fold.templates)
    n_probes = (len(FOLDS[0]) + len(HOLDOUT)) * 40
    assert fold.protected_table.scores.size == n_enrolled * n_probes
    assert set(fold.model.train_subjects).isdisjoint(FOLDS[0])
    assert set(fold.model.train_subjects).isdisjoint(HOLDOUT)
    assert np.isnan(fold.embedding_table.template_score_mean).all()
    assert np.isfinite(fold.protected_table.template_score_mean).all()
    genuine, holdout = split_scores(fold.protected_table, "impostor_holdout")
    assert genuine.mean() > holdout.mean() + 0.1


def test_a_refused_enrollment_is_reported_and_its_probes_still_count(
    folds: list[FoldScores],
) -> None:
    fold = folds[0]
    assert fold.failed_to_enroll == (SHORT_ENROLLMENT_SUBJECT,)
    table = fold.protected_table
    assert SHORT_ENROLLMENT_SUBJECT not in table.claimed_subject
    assert SHORT_ENROLLMENT_SUBJECT in table.source_subject


def test_protected_scores_use_the_claimed_identitys_key(folds: list[FoldScores]) -> None:
    fold = folds[1]
    claimed = min(fold.templates)
    table = fold.protected_table
    rows = (table.claimed_subject == claimed) & (table.source_subject == 90)
    probe = identity_feature_matrix(90, 2)
    embeddings = embed(fold.model, probe)
    n_bits = fold.model.n_components

    def under(reference: str) -> np.ndarray:
        projection = projection_matrix(derive_key(SECRET, reference, 1), n_bits, n_bits)
        return hamming_similarity(protect(embeddings, projection), fold.templates[claimed].bits)

    np.testing.assert_array_equal(table.scores[rows], under(subject_ref(claimed)))
    assert not np.array_equal(table.scores[rows], under(subject_ref(90)))


def test_threshold_selection_never_sees_the_holdout(folds: list[FoldScores]) -> None:
    table = concatenate_tables([fold.protected_table for fold in folds])
    with pytest.raises(AssertionError, match="impostor-holdout"):
        select_operating_threshold(table, 0.01)
    point = select_operating_threshold(select_rows(table, ~table.source_is_holdout), 0.01)
    assert point.far <= 0.01


def test_random_pairing_control_sits_near_chance(folds: list[FoldScores]) -> None:
    table = concatenate_tables([fold.protected_table for fold in folds])
    genuine, impostor = split_scores(table, "impostor_holdout")
    real = eer_value(equal_error_rate(error_rates(genuine, impostor)))
    control = random_pairing_control_eer(table, seed=3)
    assert real < 0.2
    assert 0.35 <= control <= 0.65


def test_cross_fitting_scores_each_fold_with_a_layer_trained_on_the_others(
    folds: list[FoldScores],
) -> None:
    protected = concatenate_tables([fold.protected_table for fold in folds])
    versions = {fold.fold_index: fold.representation_version for fold in folds}
    n_bits = folds[0].model.n_components
    decision, decision_versions = cross_fit_decision_table(
        protected, n_bits=n_bits, representation_versions=versions
    )
    assert decision.domain == "decision"
    assert len(set(decision_versions)) == len(folds)

    training = select_rows(protected, (protected.fold != 0) & ~protected.source_is_holdout)
    layer = fit_decision_layer(
        training, n_bits=n_bits, representation_versions=(versions[1], versions[2])
    )
    target = protected.fold == 0
    expected = decision_llr(
        layer,
        protected.scores[target],
        protected.template_score_mean[target],
        protected.template_score_std[target],
        protected.probe_window_ok[target],
    )
    np.testing.assert_array_equal(decision.scores[target], expected)


def test_reissued_templates_are_unlinkable(folds: list[FoldScores]) -> None:
    fold = folds[2]
    reissued = reissue_fold_templates(
        MATRICES,
        fold,
        split_kind="cross_condition",
        streaming=STREAMING,
        impostor_holdout=HOLDOUT,
        master_secret=SECRET,
        now=NOW,
    )
    assert {t.key_version for t in reissued.values()} == {2}
    mean_agreement, accepted = revocation_agreement(fold.templates, reissued, threshold=0.9)
    assert 0.35 <= mean_agreement <= 0.65
    assert accepted == 0.0
    with pytest.raises(ValueError, match="key_version"):
        revocation_agreement(fold.templates, fold.templates, threshold=0.9)


def test_temporal_split_fold_runs_end_to_end() -> None:
    """Fold 1: the 10-window recording in fold 0 is too short to split in time, and
    temporal_split refuses it loudly rather than dropping the subject."""
    assert SHORT_ENROLLMENT_SUBJECT not in FOLDS[1]
    fold = run_fold(1, split_kind="temporal")
    check_score_table(fold.protected_table, HOLDOUT)
    assert fold.protected_table.split_kind == "temporal"
    genuine, holdout = split_scores(fold.protected_table, "impostor_holdout")
    assert genuine.mean() > holdout.mean()
    with pytest.raises(ValueError, match="0 test windows"):
        run_fold(0, split_kind="temporal")
