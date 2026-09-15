"""Cohort-only score and LLR distributions, for choosing session thresholds.

Session thresholds may come only from cohort scores and genuine-only replays of enrollable
subjects (session/logic.py, D-022). This script never loads a holdout subject's EEG and never
reads a holdout artifact. It reads config/impostor_holdout.json only to take the enrollable
list and to keep holdout subjects out, which is asserted.

It recomputes the headline protocol's cross-condition fold scores for enrollable subjects:
the same folds, seed, models, and evaluation key as the verification run, so these rows are
that run's genuine and cohort-impostor rows. Genuine probes are eyes-closed windows, which is
also what a replayed genuine session streams.

- score: window-level Hamming similarity of protected templates, in steps of 1/64.
- llr: decision layer v0, cross-fitted, so each fold is scored by a layer trained on the
  other folds (D-021).

The worst decile is ranked on cohort-impostor per-subject EER, not on the holdout headline
table. A list chosen from holdout results would carry holdout information into the
thresholds.

Usage:
    python -m scripts.measurements.cohort_score_distributions
"""

import argparse
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from neuroauth.cohorts import DEFAULT_HOLDOUT_PATH, assert_holdout_excluded, load_holdout_record
from neuroauth.config import StreamingConfig
from neuroauth.dsp.io import load_baseline_recordings
from neuroauth.dsp.streaming import process_recording_bounded
from neuroauth.verification.embedding import EmbeddingConfig
from neuroauth.verification.metrics import (
    FAR_TARGETS,
    ScoreTable,
    concatenate_tables,
    eer_value,
    equal_error_rate,
    error_rates,
    frr_at_far,
    genuine_mask,
    impostor_mask,
    n_impostor_pairs,
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
from scripts.train_baseline import PreconditionError, check_holdout_committed, working_tree_changes

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLIT = "cross_condition"
PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)


def row(label: str, values: NDArray[np.float64], digits: int) -> str:
    cells = [f"{np.mean(values):.{digits}f}", f"{np.std(values):.{digits}f}"]
    cells += [f"{v:.{digits}f}" for v in np.percentile(values, PERCENTILES)]
    width = digits + 4
    return f"  {label:<34}{values.size:>8}  " + "".join(f"{c:>{width}}" for c in cells)


def header(digits: int) -> str:
    width = digits + 4
    names = ["mean", "std", *[f"p{p}" for p in PERCENTILES]]
    return f"  {'':<34}{'n':>8}  " + "".join(f"{n:>{width}}" for n in names)


def block(table: ScoreTable, title: str, worst: frozenset[int], digits: int) -> None:
    genuine = genuine_mask(table)
    impostor = impostor_mask(table, "cohort")
    in_worst = np.isin(table.claimed_subject, np.array(sorted(worst), dtype=np.int64))
    print(f"\n{title}")
    print(header(digits))
    for group, mask in (
        ("all subjects", np.ones_like(genuine)),
        ("worst decile (cohort)", in_worst),
        ("other subjects", ~in_worst),
    ):
        print(row(f"{group}: genuine", table.scores[genuine & mask], digits))
        print(row(f"{group}: impostor", table.scores[impostor & mask], digits))

    rates = error_rates(table.scores[genuine], table.scores[impostor])
    pairs = n_impostor_pairs(table, "cohort")
    eer = equal_error_rate(rates)
    print(
        f"  cohort operating points ({pairs} claimed-impostor pairs): "
        f"EER {eer_value(eer):.4f} at threshold {eer.threshold:.{digits}f} "
        f"(FAR {eer.far:.4f}, FRR {eer.frr:.4f})"
    )
    for target in FAR_TARGETS:
        point = frr_at_far(rates, target, n_impostor_pairs=pairs)
        flag = "  UNDER-RESOLVED" if point.under_resolved else ""
        print(
            f"    FAR <= {target:g}: threshold {point.threshold:.{digits}f}, "
            f"FAR {point.far:.4f}, FRR {point.frr:.4f}{flag}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cohort-only score distributions.")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    args = parser.parse_args(argv)
    holdout_path = REPO_ROOT / DEFAULT_HOLDOUT_PATH
    try:
        git_head = check_holdout_committed(holdout_path, REPO_ROOT)
        dirty = bool(working_tree_changes(REPO_ROOT))
    except PreconditionError as exc:
        print(f"refusing to run: {exc}", file=sys.stderr)
        return 2

    record = load_holdout_record(holdout_path)
    holdout = frozenset(record.impostor_holdout)
    enrollable = record.enrollable
    assert_holdout_excluded(enrollable, holdout)

    streaming = StreamingConfig()
    matrices = [
        process_recording_bounded(recording, streaming)
        for recording in load_baseline_recordings(enrollable, args.data_dir, skip_failures=False)
    ]
    assert_holdout_excluded((m.subject_id for m in matrices if m.subject_id is not None), holdout)

    folds = subject_folds(enrollable, N_FOLDS, FOLD_SEED)
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
        for index, fold in enumerate(folds)
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
    genuine = genuine_mask(protected)
    pooled = eer_value(
        equal_error_rate(
            error_rates(
                protected.scores[genuine], protected.scores[impostor_mask(protected, "cohort")]
            )
        )
    )
    worst = frozenset(summarize_tail(per_subject, pooled_eer=pooled).worst_decile_subjects)
    ranked = sorted(worst, key=lambda s: (-per_subject[s], s))

    print(f"git_head {git_head}  dirty {dirty}  split {SPLIT}  impostors: cohort only")
    print(f"enrollable subjects loaded: {len(matrices) // 2}; holdout subjects loaded: 0")
    print(
        "worst decile by cohort per-subject EER: "
        + ", ".join(f"S{s:03d} ({per_subject[s]:.3f})" for s in ranked)
    )
    block(protected, "SCORE: Hamming similarity (protected), window level", worst, 4)
    block(decision, "LLR: decision layer v0, cross-fitted, window level", worst, 3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
