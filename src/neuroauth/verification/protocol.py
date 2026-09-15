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
  threshold or to fit the decision layer. A threshold picked on H and then reported on H is
  selection on the test set.

Enrollment and probes (headline, cross-condition): enroll on the eyes-open recording and
probe with eyes-closed windows, for genuine and impostor comparisons alike. Taking impostor
probes from the same condition means a genuine comparison and an impostor comparison differ
only in whose EEG it is.

The key used for a protected comparison is always the claimed identity's. Transforming each
impostor probe with the impostor's own key is the "unstolen token" protocol error that
drives BioHash EERs toward zero in parts of the literature: the key does the discriminating,
not the biometric.
"""

import dataclasses
from dataclasses import dataclass
from datetime import datetime
from typing import Final

import numpy as np
from numpy.typing import NDArray

from neuroauth.config import StreamingConfig
from neuroauth.dsp.types import FeatureMatrix, QualityReport
from neuroauth.models.splits import temporal_split
from neuroauth.templates.cancelable import (
    derive_key,
    hamming_similarity,
    projection_matrix,
    protect,
)
from neuroauth.templates.enrollment import (
    EnrollmentRefusedError,
    ProtectedTemplate,
    enroll_subject,
    representation_version,
    revoke_and_reissue,
    subject_ref,
)
from neuroauth.verification.decision import DecisionLayer, decision_llr, fit_decision_layer
from neuroauth.verification.embedding import (
    EmbeddingConfig,
    EmbeddingModel,
    embed_values,
    fit_embedding,
)
from neuroauth.verification.metrics import (
    OperatingPoint,
    ScoreTable,
    eer_value,
    equal_error_rate,
    error_rates,
    frr_at_far,
    genuine_mask,
    n_impostor_pairs,
    select_rows,
    split_scores,
)

N_FOLDS: Final = 5
FOLD_SEED: Final = 20260915
TEMPORAL_ENROLL_FRACTION: Final = 0.7
"""Temporal split only: enroll on the leading 70% of eyes-open, probe after a one-window
guard. Same value as Phase 1's TRAIN_FRACTION. All three fixed before any Phase 2 result."""

THRESHOLD_SELECTION_FAR: Final = 0.01
"""The deployed window-level threshold is the one achieving FAR <= 0.01 on cohort impostor
scores. It is reported on the holdout as VerificationReport.deployed."""

SPLIT_KINDS: Final = ("cross_condition", "temporal")


def subject_folds(
    subjects: tuple[int, ...], n_folds: int, seed: int
) -> tuple[tuple[int, ...], ...]:
    """Deal subjects into subject-disjoint folds.

    A seeded permutation of the sorted ids, dealt round-robin, sorted within each fold. The
    fold record is written into the run summary. It is not a committed file like the
    holdout, because folds are not a security boundary: every enrollable subject is
    evaluated in exactly one of them.

    Returns:
        n_folds tuples that partition subjects, with sizes differing by at most one.

    Raises:
        ValueError: On duplicate ids, or n_folds below 2 or above len(subjects).
    """
    ids = sorted(int(subject) for subject in subjects)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate subject ids")
    if not 2 <= n_folds <= len(ids):
        raise ValueError(f"n_folds must be in [2, {len(ids)}], got {n_folds}")
    permuted = np.random.default_rng(seed).permutation(np.asarray(ids, dtype=np.int64))
    return tuple(tuple(sorted(int(s) for s in permuted[k::n_folds])) for k in range(n_folds))


def _row_offsets(matrices: list[FeatureMatrix]) -> NDArray[np.int64]:
    sizes = np.array([matrix.values.shape[0] for matrix in matrices], dtype=np.int64)
    return np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(sizes, dtype=np.int64)))


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
        ValueError: On an unknown split_kind, a matrix without subject_id, or a subject
            missing a required recording.
        AssertionError: If any probe window shares samples with an enrollment window of the
            same recording.
    """
    offsets = _row_offsets(matrices)
    conditions: dict[int, set[str]] = {}
    for matrix in matrices:
        if matrix.subject_id is None or matrix.condition is None:
            raise ValueError("every matrix needs subject_id and condition")
        conditions.setdefault(matrix.subject_id, set()).add(matrix.condition)

    def rows_where(condition: str) -> NDArray[np.int64]:
        parts = [
            np.arange(offsets[k], offsets[k + 1], dtype=np.int64)
            for k, matrix in enumerate(matrices)
            if matrix.condition == condition
        ]
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)

    if split_kind == "cross_condition":
        missing = sorted(s for s, c in conditions.items() if not {"eyes_open", "eyes_closed"} <= c)
        if missing:
            raise ValueError(f"subjects missing an eyes-open or eyes-closed recording: {missing}")
        return rows_where("eyes_open"), rows_where("eyes_closed")
    if split_kind == "temporal":
        missing = sorted(s for s, c in conditions.items() if "eyes_open" not in c)
        if missing:
            raise ValueError(f"subjects missing an eyes-open recording: {missing}")
        positions = [k for k, matrix in enumerate(matrices) if matrix.condition == "eyes_open"]
        subset = [matrices[k] for k in positions]
        enroll, probe = temporal_split(
            subset, TEMPORAL_ENROLL_FRACTION, window_s=window_s, guard_s=window_s
        )
        subset_offsets = _row_offsets(subset)
        global_starts = offsets[np.asarray(positions, dtype=np.int64)]

        def to_global(rows: NDArray[np.int64]) -> NDArray[np.int64]:
            which = np.searchsorted(subset_offsets, rows, side="right") - 1
            return (rows - subset_offsets[which] + global_starts[which]).astype(np.int64)

        return to_global(enroll), to_global(probe)
    raise ValueError(f"unknown split_kind {split_kind!r}")


def _select_matrix_rows(matrix: FeatureMatrix, rows: NDArray[np.int64]) -> FeatureMatrix:
    return dataclasses.replace(
        matrix,
        values=matrix.values[rows],
        quality=QualityReport(
            window_ok=matrix.quality.window_ok[rows],
            channel_ok=matrix.quality.channel_ok[rows],
            flags=tuple(matrix.quality.flags[i] for i in rows.tolist()),
        ),
        onsets_s=matrix.onsets_s[rows],
    )


def _enrollment_features(
    matrices: list[FeatureMatrix],
    offsets: NDArray[np.int64],
    rows: NDArray[np.int64],
) -> FeatureMatrix:
    """The enrollment rows of one subject, which always come from a single recording."""
    which = np.unique(np.searchsorted(offsets, rows, side="right") - 1)
    if which.size != 1:
        raise ValueError("enrollment rows span more than one recording")
    k = int(which[0])
    return _select_matrix_rows(matrices[k], rows - offsets[k])


@dataclass(frozen=True)
class FoldScores:
    """Everything one fold produced. Score tables, templates, and population statistics only.

    Attributes:
        fold_index: Recorded in the tables' fold column.
        embedding_table: Cosine similarity of unprotected embeddings (in memory only until
            reduced to these scores).
        protected_table: Hamming similarity of protected bits.
        templates: Enrolled subject -> template.
        failed_to_enroll: Fold subjects whose enrollment was refused.
        model: The fold's embedding, fitted without the fold or the holdout.
        representation_version: Of the fold's templates.
    """

    fold_index: int
    embedding_table: ScoreTable
    protected_table: ScoreTable
    templates: dict[int, ProtectedTemplate]
    failed_to_enroll: tuple[int, ...]
    model: EmbeddingModel
    representation_version: str


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
    enrolled_at: datetime,
) -> FoldScores:
    """Fit the fold's embedding, enroll the fold, and score every comparison in both domains.

    Each fold-k subject s is enrolled from its enrollment rows, then scored against its own
    probes (genuine), every other fold-k subject's probes (cohort impostors), and every
    holdout subject's probes (holdout impostors). Protected scores use s's key for every
    comparison against s.

    Enrollment embeddings and per-window embeddings are locals. The function returns score
    tables, templates, and the fold's population-statistics model only.

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
        enrolled_at: Recorded on the templates.

    Returns:
        The fold's scores and templates.

    Raises:
        AssertionError: If train and fold intersect, train and H intersect, fold and H
            intersect, or a matrix's subject is in none of the three (hard rules 1 and 5).
    """
    for name, first, second in (
        ("training and fold", train_subjects, fold_subjects),
        ("training and holdout", train_subjects, impostor_holdout),
        ("fold and holdout", fold_subjects, impostor_holdout),
    ):
        both = sorted(first & second)
        if both:
            raise AssertionError(f"{name} subjects overlap: {both}")
    present = {matrix.subject_id for matrix in matrices}
    if None in present:
        raise ValueError("every matrix needs a subject_id")
    stray = sorted(
        s
        for s in present
        if s is not None and s not in train_subjects | fold_subjects | impostor_holdout
    )
    if stray:
        raise AssertionError(f"subjects in no cohort of this fold: {stray}")

    fingerprint = streaming.fingerprint()
    model = fit_embedding(
        [m for m in matrices if m.subject_id in train_subjects],
        config=embedding_config,
        streaming_fingerprint=fingerprint,
        impostor_holdout=impostor_holdout,
        evaluated_subjects=fold_subjects,
    )
    representation = representation_version(model)

    scored = [m for m in matrices if m.subject_id in fold_subjects | impostor_holdout]
    offsets = _row_offsets(scored)
    labels = np.concatenate(
        [np.full(m.values.shape[0], m.subject_id, dtype=np.int64) for m in scored]
    )
    enroll_rows, probe_rows = enrollment_and_probe_rows(
        scored, split_kind, window_s=streaming.pipeline.window.window_s
    )

    probe_values = np.concatenate([m.values for m in scored])[probe_rows]
    probe_embeddings = embed_values(model, probe_values)
    probe_norms = np.linalg.norm(probe_embeddings, axis=1)
    probe_sources = labels[probe_rows]
    probe_ok = np.concatenate([m.quality.window_ok for m in scored])[probe_rows]
    probe_onsets = np.concatenate([m.onsets_s for m in scored])[probe_rows]
    probe_is_holdout = np.isin(probe_sources, np.array(sorted(impostor_holdout), dtype=np.int64))

    templates: dict[int, ProtectedTemplate] = {}
    failed: list[int] = []
    columns: dict[str, list[NDArray[np.generic]]] = {
        name: [] for name in ("embedding", "protected", "claimed", "template_mean", "template_std")
    }
    for subject in sorted(fold_subjects):
        subject_rows = enroll_rows[labels[enroll_rows] == subject]
        if subject_rows.size == 0:
            raise ValueError(f"fold subject {subject} has no enrollment rows")
        features = _enrollment_features(scored, offsets, subject_rows)
        try:
            template = enroll_subject(
                features,
                model,
                master_secret,
                subject_id=subject,
                subject_ref=subject_ref(subject),
                impostor_holdout=impostor_holdout,
                now=enrolled_at,
            )
        except EnrollmentRefusedError:
            failed.append(subject)
            continue
        templates[subject] = template

        center = embed_values(model, features.values).mean(axis=0)
        denominator = np.maximum(probe_norms * np.linalg.norm(center), np.finfo(np.float64).tiny)
        cosine = (probe_embeddings @ center) / denominator
        key = derive_key(master_secret, template.subject_ref, template.key_version)
        projection = projection_matrix(key, model.n_components, model.n_components)
        similarity = hamming_similarity(protect(probe_embeddings, projection), template.bits)

        n_probes = probe_rows.size
        columns["embedding"].append(cosine)
        columns["protected"].append(similarity)
        columns["claimed"].append(np.full(n_probes, subject, dtype=np.int64))
        columns["template_mean"].append(np.full(n_probes, template.enrollment_score_mean))
        columns["template_std"].append(np.full(n_probes, template.enrollment_score_std))

    n_enrolled = len(templates)
    if n_enrolled == 0:
        raise ValueError(f"fold {fold_index}: no subject enrolled")

    def table(scores_key: str, domain: str, with_stats: bool) -> ScoreTable:
        n_rows = n_enrolled * probe_rows.size
        return ScoreTable(
            scores=np.concatenate(columns[scores_key]).astype(np.float64),
            claimed_subject=np.concatenate(columns["claimed"]).astype(np.int64),
            source_subject=np.tile(probe_sources, n_enrolled),
            source_is_holdout=np.tile(probe_is_holdout, n_enrolled),
            fold=np.full(n_rows, fold_index, dtype=np.int64),
            probe_onset_s=np.tile(probe_onsets, n_enrolled),
            probe_window_ok=np.tile(probe_ok, n_enrolled),
            template_score_mean=(
                np.concatenate(columns["template_mean"]).astype(np.float64)
                if with_stats
                else np.full(n_rows, np.nan)
            ),
            template_score_std=(
                np.concatenate(columns["template_std"]).astype(np.float64)
                if with_stats
                else np.full(n_rows, np.nan)
            ),
            domain=domain,  # type: ignore[arg-type]
            split_kind=split_kind,
        )

    return FoldScores(
        fold_index=fold_index,
        embedding_table=table("embedding", "embedding", with_stats=False),
        protected_table=table("protected", "protected", with_stats=True),
        templates=templates,
        failed_to_enroll=tuple(failed),
        model=model,
        representation_version=representation,
    )


def reissue_fold_templates(
    matrices: list[FeatureMatrix],
    fold: FoldScores,
    *,
    split_kind: str,
    streaming: StreamingConfig,
    impostor_holdout: frozenset[int],
    master_secret: bytes,
    now: datetime,
) -> dict[int, ProtectedTemplate]:
    """Revoke every template of a fold and reissue it under key_version + 1.

    The same enrollment windows are used, because eegmmidb has one session per subject; the
    unlinkability measured therefore comes entirely from the key.
    """
    scored = [m for m in matrices if m.subject_id in fold.templates]
    offsets = _row_offsets(scored)
    labels = np.concatenate(
        [np.full(m.values.shape[0], m.subject_id, dtype=np.int64) for m in scored]
    )
    enroll_rows, _ = enrollment_and_probe_rows(
        scored, split_kind, window_s=streaming.pipeline.window.window_s
    )
    reissued: dict[int, ProtectedTemplate] = {}
    for subject, old in fold.templates.items():
        features = _enrollment_features(
            scored, offsets, enroll_rows[labels[enroll_rows] == subject]
        )
        _, template = revoke_and_reissue(
            old,
            features,
            fold.model,
            master_secret,
            subject_id=subject,
            impostor_holdout=impostor_holdout,
            reason="evaluation: revocation unlinkability check",
            actor="script:evaluate_verification",
            now=now,
        )
        reissued[subject] = template
    return reissued


def select_operating_threshold(cohort_table: ScoreTable, target_far: float) -> OperatingPoint:
    """Choose a window-level threshold from cohort impostor scores only.

    Raises:
        AssertionError: If any row's source is a holdout subject. Thresholds never see H.
    """
    if cohort_table.source_is_holdout.any():
        raise AssertionError("impostor-holdout rows in threshold-selection data")
    genuine, impostor = split_scores(cohort_table, "cohort")
    return frr_at_far(
        error_rates(genuine, impostor),
        target_far,
        n_impostor_pairs=n_impostor_pairs(cohort_table, "cohort"),
    )


def _derangement(n: int, rng: np.random.Generator) -> NDArray[np.int64]:
    identity = np.arange(n)
    while True:
        permutation = rng.permutation(n)
        if not np.any(permutation == identity):
            return permutation.astype(np.int64)


def random_pairing_control_eer(table: ScoreTable, *, seed: int) -> float:
    """Pooled EER with every genuine comparison replaced by a comparison with someone else.

    For each claimed subject s, the genuine scores are replaced by s's existing
    cohort-impostor scores against one other enrolled subject of the same fold, chosen by a
    seeded derangement. Those rows leave the impostor side, and every other impostor row
    stays. Both sides are then impostor comparisons, so EER must sit near 0.5.

    What it catches: pairing and metric bugs that make "genuine" mean something other than
    same identity, such as a template built from the probe windows or misaligned subject
    columns. What it cannot catch: leakage into the embedding fit. That is ruled out
    structurally by the disjointness assertions in score_fold, as D-007 separates the
    shuffled-label control from assert_no_window_overlap.

    Raises:
        ValueError: If a fold has fewer than two enrolled subjects.
    """
    rng = np.random.default_rng(seed)
    genuine = genuine_mask(table)
    substitute = np.zeros(table.scores.shape[0], dtype=np.bool_)
    for fold in np.unique(table.fold).tolist():
        in_fold = table.fold == fold
        enrolled = np.unique(table.claimed_subject[genuine & in_fold])
        if enrolled.size < 2:
            raise ValueError(f"fold {fold} has fewer than two enrolled subjects")
        partner = enrolled[_derangement(enrolled.size, rng)]
        for claimed, other in zip(enrolled.tolist(), partner.tolist(), strict=True):
            substitute |= (
                in_fold & (table.claimed_subject == claimed) & (table.source_subject == other)
            )
    impostor = ~genuine & ~substitute
    return eer_value(
        equal_error_rate(error_rates(table.scores[substitute], table.scores[impostor]))
    )


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
        ValueError: If the subject sets differ or are empty, or any pair shares a key_version.
    """
    if not revoked or set(revoked) != set(reissued):
        raise ValueError("revoked and reissued templates must cover the same, non-empty subjects")
    agreements = []
    for subject, old in revoked.items():
        new = reissued[subject]
        if old.key_version == new.key_version:
            raise ValueError(f"subject {subject}: revoked and reissued share key_version")
        agreements.append(float(hamming_similarity(old.bits[None, :], new.bits)[0]))
    values = np.array(agreements)
    return float(values.mean()), float(np.mean(values >= threshold))


def cross_fit_decision_layers(
    protected: ScoreTable,
    *,
    n_bits: int,
    representation_versions: dict[int, str],
) -> dict[int, DecisionLayer]:
    """One decision layer per fold, each trained on the other folds' rows.

    Each layer trains on the other folds' genuine and cohort-impostor rows only. Holdout rows
    are excluded before fitting, and fit_decision_layer asserts it.

    Returns:
        Fold -> the layer that scores that fold, in ascending fold order.

    Raises:
        ValueError: If the table is not protected-domain or has fewer than two folds.
    """
    if protected.domain != "protected":
        raise ValueError(f"cross-fitting needs protected scores, got {protected.domain}")
    folds = np.unique(protected.fold).tolist()
    if len(folds) < 2:
        raise ValueError("cross-fitting needs at least two folds")
    return {
        fold: fit_decision_layer(
            select_rows(protected, (protected.fold != fold) & ~protected.source_is_holdout),
            n_bits=n_bits,
            representation_versions=tuple(
                representation_versions[other] for other in folds if other != fold
            ),
        )
        for fold in folds
    }


def cross_fit_decision_table(
    protected: ScoreTable,
    *,
    n_bits: int,
    representation_versions: dict[int, str],
) -> tuple[ScoreTable, tuple[str, ...]]:
    """Decision-domain scores, cross-fitted: fold k is scored by a layer trained on the others.

    No row is ever scored by a layer that saw it (cross_fit_decision_layers).

    Returns:
        (decision_table, decision_versions), with one version per fold in ascending fold order.

    Raises:
        ValueError: If the table is not protected-domain or has fewer than two folds.
    """
    layers = cross_fit_decision_layers(
        protected, n_bits=n_bits, representation_versions=representation_versions
    )
    llr = np.full(protected.scores.shape[0], np.nan)
    for fold, layer in layers.items():
        target = protected.fold == fold
        llr[target] = decision_llr(
            layer,
            protected.scores[target],
            protected.template_score_mean[target],
            protected.template_score_std[target],
            protected.probe_window_ok[target],
        )
    versions = tuple(layer.decision_version for layer in layers.values())
    return dataclasses.replace(protected, scores=llr, domain="decision"), versions
