"""Cohort-only score tables, shared by the measurements used to choose session parameters.

Session parameters may come only from cohort scores and genuine-only replays of enrollable
subjects (session/logic.py, D-022). Nothing here loads a holdout subject's EEG or reads a
holdout artifact. config/impostor_holdout.json is read only to take the enrollable list and
to keep holdout subjects out, and both are asserted.

The headline protocol's cross-condition fold scores are recomputed for enrollable subjects,
with the same folds, seed, models, and evaluation key as the verification run, so these rows
are that run's genuine and cohort-impostor rows. Genuine probes are eyes-closed windows, which
is what a replayed genuine session streams.

The worst decile is ranked on cohort-impostor per-subject EER, not on the holdout headline
table. A list chosen from holdout results would carry holdout information into the session
parameters.

Dirty-tree rule: numbers from uncommitted code are printed under a NOT CITABLE banner. Commit,
then re-run before citing them.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from neuroauth.cohorts import DEFAULT_HOLDOUT_PATH, assert_holdout_excluded, load_holdout_record
from neuroauth.config import StreamingConfig
from neuroauth.dsp.io import load_baseline_recordings
from neuroauth.dsp.streaming import process_recording_bounded
from neuroauth.verification.embedding import EmbeddingConfig
from neuroauth.verification.metrics import (
    ScoreTable,
    concatenate_tables,
    eer_value,
    equal_error_rate,
    error_rates,
    genuine_mask,
    impostor_mask,
    per_subject_eer,
    summarize_tail,
)
from neuroauth.verification.protocol import (
    FOLD_SEED,
    N_FOLDS,
    cross_fit_decision_table,
    score_fold,
    subject_folds,
)
from scripts.evaluate_verification import EVALUATION_MASTER_SECRET
from scripts.train_baseline import check_holdout_committed, working_tree_changes

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLIT = "cross_condition"
NOT_CITABLE = "NOT CITABLE: produced by uncommitted code. Commit, then re-run before citing."


@dataclass(frozen=True)
class CohortScores:
    """Genuine and cohort-impostor rows only, in two score domains.

    Attributes:
        git_head: HEAD when the tables were computed.
        dirty: Uncommitted changes outside artifacts/ at the time.
        streaming: Settings the features were built with.
        protected: Hamming similarity of protected templates.
        decision: Decision layer v0 LLR, cross-fitted across folds.
        per_subject_eer: Claimed subject -> EER against cohort impostors.
        worst_decile: Worst decile by that EER, worst first.
        n_subjects_loaded: Enrollable subjects whose EEG was loaded.
    """

    git_head: str
    dirty: bool
    streaming: StreamingConfig
    protected: ScoreTable
    decision: ScoreTable
    per_subject_eer: dict[int, float]
    worst_decile: tuple[int, ...]
    n_subjects_loaded: int


def compute_cohort_scores(data_dir: Path) -> CohortScores:
    """Recompute the cohort-only cross-condition tables.

    Raises:
        PreconditionError: If the holdout record is not committed or git is unavailable.
        AssertionError: If any holdout subject is loaded or appears in a table.
    """
    holdout_path = REPO_ROOT / DEFAULT_HOLDOUT_PATH
    git_head = check_holdout_committed(holdout_path, REPO_ROOT)
    dirty = bool(working_tree_changes(REPO_ROOT))

    record = load_holdout_record(holdout_path)
    holdout = frozenset(record.impostor_holdout)
    enrollable = record.enrollable
    assert_holdout_excluded(enrollable, holdout)

    streaming = StreamingConfig()
    matrices = [
        process_recording_bounded(recording, streaming)
        for recording in load_baseline_recordings(enrollable, data_dir, skip_failures=False)
    ]
    assert_holdout_excluded((m.subject_id for m in matrices if m.subject_id is not None), holdout)

    enrolled_at = datetime.now(UTC)
    fold_scores = [
        score_fold(
            matrices,
            fold_index=index,
            fold_subjects=frozenset(fold),
            train_subjects=frozenset(enrollable) - frozenset(fold),
            impostor_holdout=holdout,
            split_kind=SPLIT,
            streaming=streaming,
            embedding_config=EmbeddingConfig(),
            master_secret=EVALUATION_MASTER_SECRET,
            enrolled_at=enrolled_at,
        )
        for index, fold in enumerate(subject_folds(enrollable, N_FOLDS, FOLD_SEED))
    ]
    protected = concatenate_tables([f.protected_table for f in fold_scores])
    if protected.source_is_holdout.any():
        raise AssertionError("holdout rows present in a cohort-only table")
    decision, _ = cross_fit_decision_table(
        protected,
        n_bits=min(f.model.n_components for f in fold_scores),
        representation_versions={f.fold_index: f.representation_version for f in fold_scores},
    )

    per_subject = per_subject_eer(protected, "cohort")
    pooled = eer_value(
        equal_error_rate(
            error_rates(
                protected.scores[genuine_mask(protected)],
                protected.scores[impostor_mask(protected, "cohort")],
            )
        )
    )
    tail = summarize_tail(per_subject, pooled_eer=pooled)
    return CohortScores(
        git_head=git_head,
        dirty=dirty,
        streaming=streaming,
        protected=protected,
        decision=decision,
        per_subject_eer=per_subject,
        worst_decile=tail.worst_decile_subjects,
        n_subjects_loaded=len({m.subject_id for m in matrices}),
    )


def provenance_lines(scores: CohortScores) -> list[str]:
    lines = []
    if scores.dirty:
        lines.append(NOT_CITABLE)
    lines.append(
        f"git_head {scores.git_head}  dirty {scores.dirty}  split {SPLIT}  impostors: cohort only"
    )
    lines.append(
        f"enrollable subjects loaded: {scores.n_subjects_loaded}; holdout subjects loaded: 0"
    )
    lines.append(
        "worst decile by cohort per-subject EER: "
        + ", ".join(f"S{s:03d} ({scores.per_subject_eer[s]:.3f})" for s in scores.worst_decile)
    )
    return lines
