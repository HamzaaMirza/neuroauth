"""Run the Phase 2 open-set verification evaluation and write the committed artifacts.

Everything below was fixed before any Phase 2 result (D-016 rule; D-020, D-021):

    splits      cross_condition (headline: enroll eyes-open, probe eyes-closed), temporal
    domains     protected (headline), embedding (cost of protection), decision (v0, cross-fitted)
    impostors   the committed holdout (reporting only) and cohort impostors (thresholds only)
    folds       N_FOLDS subject-disjoint folds of the enrollable cohort, seed FOLD_SEED
    headline    FRR at FAR = 0.01, cross_condition, protected, holdout impostors

Gates, all checked before anything is written:
    - config/impostor_holdout.json is committed and unmodified; the list is loaded from it
      (D-008). Writing artifacts/ requires a clean working tree.
    - every score table passes check_score_table (no holdout subject claimed, cohort flags
      consistent); every fold asserts training/fold/holdout disjointness (hard rules 1, 5).
    - every random-pairing control EER is at least RANDOM_PAIRING_CONTROL_MIN_EER.

Judged, with a priori criteria:
    - P1a heavy per-subject tail and P1b concordance with Phase 1 F1 (headline table only)
    - material cost of protection (protected minus embedding EER)
    - revocation: every template reissued under key_version 2 from the same enrollment
      windows, bit agreement and acceptance at the cohort-selected threshold

The master secret is a fixed, public, evaluation-only value. The evaluation measures the
protocol, not key secrecy, and the value is never accepted as a deployment secret.

Artifacts hold scores, rates, counts, and versions only (hard rule 3).

Usage:
    python -m scripts.evaluate_verification
    python -m scripts.evaluate_verification --subjects 15 --impostors 5 --out <dir>  # partial
"""

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mne
import numpy as np
import scipy
import sklearn

from neuroauth.cohorts import DEFAULT_HOLDOUT_PATH, assert_holdout_excluded, load_holdout_record
from neuroauth.config import StreamingConfig
from neuroauth.dsp.io import load_baseline_recordings
from neuroauth.dsp.streaming import check_margins, process_recording_bounded
from neuroauth.dsp.types import FeatureMatrix
from neuroauth.templates.cancelable import TRANSFORM_VERSION
from neuroauth.verification.decision import DECISION_ALGORITHM
from neuroauth.verification.embedding import EMBEDDING_ALGORITHM, EmbeddingConfig
from neuroauth.verification.metrics import (
    BOOTSTRAP_SEED,
    FAR_TARGETS,
    HEADLINE_FAR,
    MIN_EXPECTED_ERRORS,
    N_BOOTSTRAP,
    PAIRING_CONTROL_SEED,
    PROTECTION_MATERIAL_EER_COST,
    RANDOM_PAIRING_CONTROL_MIN_EER,
    REVOCATION_AGREEMENT_BAND,
    REVOCATION_MAX_ACCEPTED_FRACTION,
    ErrorRates,
    OperatingPoint,
    ScoreTable,
    VerificationReport,
    check_random_pairing_control,
    concatenate_tables,
    eer_value,
    error_rates,
    evaluate_verification,
    operating_point_json,
    phase1_concordance,
    select_rows,
    split_scores,
    write_det_plot,
    write_verification_report,
)
from neuroauth.verification.protocol import (
    FOLD_SEED,
    N_FOLDS,
    SPLIT_KINDS,
    THRESHOLD_SELECTION_FAR,
    FoldScores,
    cross_fit_decision_table,
    random_pairing_control_eer,
    reissue_fold_templates,
    revocation_agreement,
    score_fold,
    select_operating_threshold,
    subject_folds,
)
from scripts.train_baseline import PreconditionError, check_holdout_committed, working_tree_changes

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "artifacts" / "verification"
PHASE1_PER_CLASS_F1 = REPO_ROOT / "artifacts" / "relative_cross_condition_per_class_f1.csv"
HEADLINE_SPLIT = "cross_condition"
HEADLINE_DOMAIN = "protected"
DOMAINS = ("protected", "embedding", "decision")
EVALUATION_MASTER_SECRET = hashlib.sha256(
    b"neuroauth evaluation-only master secret; public; never a deployment key"
).digest()


def load_phase1_f1(path: Path) -> dict[int, float]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {int(row["subject_id"]): float(row["f1"]) for row in csv.DictReader(handle)}


def evaluate_split(
    matrices: list[FeatureMatrix],
    split_kind: str,
    *,
    folds: tuple[tuple[int, ...], ...],
    holdout: frozenset[int],
    streaming: StreamingConfig,
    enrolled_at: datetime,
    bootstrap: bool,
) -> dict[str, Any]:
    """Score every fold, cross-fit the decision layer, and build every report for one split."""
    enrollable = frozenset(s for fold in folds for s in fold)
    fold_scores: list[FoldScores] = []
    for index, fold in enumerate(folds):
        print(f"  {split_kind}: fold {index + 1}/{len(folds)}", flush=True)
        fold_scores.append(
            score_fold(
                matrices,
                fold_index=index,
                fold_subjects=frozenset(fold),
                train_subjects=enrollable - frozenset(fold),
                impostor_holdout=holdout,
                split_kind=split_kind,
                streaming=streaming,
                embedding_config=EmbeddingConfig(),
                master_secret=EVALUATION_MASTER_SECRET,
                enrolled_at=enrolled_at,
            )
        )
    representations = {f.fold_index: f.representation_version for f in fold_scores}
    n_bits = min(f.model.n_components for f in fold_scores)
    protected = concatenate_tables([f.protected_table for f in fold_scores])
    tables: dict[str, ScoreTable] = {
        "protected": protected,
        "embedding": concatenate_tables([f.embedding_table for f in fold_scores]),
    }
    tables["decision"], decision_versions = cross_fit_decision_table(
        protected, n_bits=n_bits, representation_versions=representations
    )
    n_failed = sum(len(f.failed_to_enroll) for f in fold_scores)

    reports: dict[str, VerificationReport] = {}
    thresholds: dict[str, OperatingPoint] = {}
    curves: dict[str, ErrorRates] = {}
    for domain in DOMAINS:
        table = tables[domain]
        cohort_rows = select_rows(table, ~table.source_is_holdout)
        selected = select_operating_threshold(cohort_rows, THRESHOLD_SELECTION_FAR)
        thresholds[domain] = selected
        control = random_pairing_control_eer(table, seed=PAIRING_CONTROL_SEED)
        common = {
            "impostor_holdout": holdout,
            "random_pairing_control_eer": control,
            "n_failure_to_enroll": n_failed,
            "representation_versions": tuple(representations[k] for k in sorted(representations)),
            "decision_versions": decision_versions if domain == "decision" else (),
            "streaming_fingerprint": streaming.fingerprint(),
        }
        deployed = selected.threshold if math.isfinite(selected.threshold) else None
        reports[f"{domain}_impostor_holdout"] = evaluate_verification(
            table,
            impostor_source="impostor_holdout",
            deployed_threshold=deployed,
            bootstrap=bootstrap and domain == HEADLINE_DOMAIN,
            **common,
        )
        reports[f"{domain}_cohort"] = evaluate_verification(
            table, impostor_source="cohort", deployed_threshold=None, bootstrap=False, **common
        )
        genuine, impostor = split_scores(table, "impostor_holdout")
        curves[domain] = error_rates(genuine, impostor)
    for report in reports.values():
        check_random_pairing_control(report)

    revoked = {s: t for f in fold_scores for s, t in f.templates.items()}
    reissued: dict[int, Any] = {}
    for f in fold_scores:
        reissued.update(
            reissue_fold_templates(
                matrices,
                f,
                split_kind=split_kind,
                streaming=streaming,
                impostor_holdout=holdout,
                master_secret=EVALUATION_MASTER_SECRET,
                now=enrolled_at,
            )
        )
    mean_agreement, accepted = revocation_agreement(
        revoked, reissued, thresholds["protected"].threshold
    )
    return {
        "reports": reports,
        "thresholds": thresholds,
        "curves": curves,
        "revocation": {
            "mean_bit_agreement": mean_agreement,
            "fraction_accepted_at_cohort_threshold": accepted,
            "agreement_band": list(REVOCATION_AGREEMENT_BAND),
            "max_accepted_fraction": REVOCATION_MAX_ACCEPTED_FRACTION,
            "works": REVOCATION_AGREEMENT_BAND[0] <= mean_agreement <= REVOCATION_AGREEMENT_BAND[1]
            and accepted <= REVOCATION_MAX_ACCEPTED_FRACTION,
        },
        "failed_to_enroll": sorted(s for f in fold_scores for s in f.failed_to_enroll),
        "representation_versions": representations,
        "decision_versions": list(decision_versions),
        "embedding_versions": {f.fold_index: f.model.embedding_version for f in fold_scores},
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run(
    matrices: list[FeatureMatrix],
    *,
    enrollable: tuple[int, ...],
    holdout: frozenset[int],
    out_dir: Path,
    streaming: StreamingConfig,
    bootstrap: bool,
    partial: bool,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate both splits, apply every gate, and only then write artifacts."""
    assert_holdout_excluded(enrollable, holdout)
    folds = subject_folds(enrollable, N_FOLDS, FOLD_SEED)
    enrolled_at = datetime.now(UTC)
    results = {
        split: evaluate_split(
            matrices,
            split,
            folds=folds,
            holdout=holdout,
            streaming=streaming,
            enrolled_at=enrolled_at,
            bootstrap=bootstrap,
        )
        for split in SPLIT_KINDS
    }

    headline_report = results[HEADLINE_SPLIT]["reports"][f"{HEADLINE_DOMAIN}_impostor_holdout"]
    per_subject = {row.subject_id: row.eer for row in headline_report.per_subject}
    concordance = phase1_concordance(per_subject, load_phase1_f1(PHASE1_PER_CLASS_F1))
    predictions = {
        "table": f"{HEADLINE_SPLIT} x {HEADLINE_DOMAIN} x impostor_holdout",
        "partial_run": partial,
        "P1a_heavy_tail": {
            "verdict": "confirmed" if headline_report.tail.heavy_tail else "refuted",
            "median_per_subject_eer": headline_report.tail.median,
            "worst_decile_mean_eer": headline_report.tail.worst_decile_mean,
            "worst_decile_subjects": list(headline_report.tail.worst_decile_subjects),
        },
        "P1b_concordance_with_phase1_f1": {
            **asdict(concordance),
            "caveat": (
                "Both phases score the same recordings, so a subject with a poor eyes-closed "
                "recording can land in both tails without identity features being the reason."
            ),
        },
        "P2_emg_cross_session": "not testable on single-session data; nothing here bears on it",
    }

    # Every gate has passed. Only now does anything touch disk.
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_splits: dict[str, Any] = {}
    for split, result in results.items():
        for name, report in result["reports"].items():
            write_verification_report(report, out_dir, prefix=f"{split}_{name}")
        holdout_reports = {d: result["reports"][f"{d}_impostor_holdout"] for d in DOMAINS}
        write_det_plot(
            result["curves"],
            out_dir / f"{split}_det.png",
            title=f"DET: {split.replace('_', '-')} split, impostor holdout",
            subtitle=(
                f"{len(enrollable)} enrolled subjects, {len(holdout)} held-out impostors, "
                f"{N_FOLDS} subject-disjoint folds. Marker at FAR {HEADLINE_FAR:.0%}."
            ),
            eer_points={d: r.eer for d, r in holdout_reports.items()},
        )
        protection_cost = eer_value(holdout_reports["protected"].eer) - eer_value(
            holdout_reports["embedding"].eer
        )
        summary_splits[split] = {
            "headline_frr_at_far": operating_point_json(holdout_reports["protected"].headline),
            "eer": {d: eer_value(r.eer) for d, r in holdout_reports.items()},
            "eer_ci95_protected": holdout_reports["protected"].eer_ci95,
            "frr_at_far": {
                d: [operating_point_json(p) for p in r.frr_at_far]
                for d, r in holdout_reports.items()
            },
            "deployed_at_cohort_threshold": {
                d: operating_point_json(r.deployed) if r.deployed is not None else None
                for d, r in holdout_reports.items()
            },
            "cohort_selected_thresholds": {
                d: operating_point_json(p) for d, p in result["thresholds"].items()
            },
            "random_pairing_control_eer": {
                name: r.random_pairing_control_eer for name, r in result["reports"].items()
            },
            "protection_cost_eer": protection_cost,
            "protection_cost_material": protection_cost > PROTECTION_MATERIAL_EER_COST,
            "revocation": result["revocation"],
            "failed_to_enroll": result["failed_to_enroll"],
            "n_impostor_pairs": holdout_reports["protected"].n_impostor_pairs,
            "representation_versions": result["representation_versions"],
            "embedding_versions": result["embedding_versions"],
            "decision_versions": result["decision_versions"],
        }

    summary = {
        **provenance,
        "what_this_is": (
            "Phase 2 open-set verification, cross-condition, protected domain, held-out "
            "impostors. Reported headline: EER with its FAR/FRR (changed after the first run, "
            "D-024); the registered headline, FRR at FAR = 0.01, is reported beside it. "
            "FAR = 0.001 is flagged under-resolved. See docs/DECISIONS.md D-020 to D-024."
        ),
        "partial_run": partial,
        "enrollable_subjects": list(enrollable),
        "impostor_holdout_used": sorted(holdout),
        "folds": [list(fold) for fold in folds],
        "splits": summary_splits,
        "predictions": predictions,
        "algorithms": {
            "embedding": EMBEDDING_ALGORITHM,
            "transform": TRANSFORM_VERSION,
            "decision": DECISION_ALGORITHM,
        },
        "streaming_fingerprint": streaming.fingerprint(),
        "a_priori": {
            "far_targets": list(FAR_TARGETS),
            "headline_far": HEADLINE_FAR,
            "min_expected_errors": MIN_EXPECTED_ERRORS,
            "threshold_selection_far": THRESHOLD_SELECTION_FAR,
            "random_pairing_control_min_eer": RANDOM_PAIRING_CONTROL_MIN_EER,
            "protection_material_eer_cost": PROTECTION_MATERIAL_EER_COST,
            "n_folds": N_FOLDS,
            "fold_seed": FOLD_SEED,
            "pairing_control_seed": PAIRING_CONTROL_SEED,
            "n_bootstrap": N_BOOTSTRAP,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
    }
    _write_json(out_dir / "run_summary.json", summary)
    return summary


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 2 verification evaluation.")
    parser.add_argument("--subjects", type=int, default=None, help="first N enrollable subjects")
    parser.add_argument("--impostors", type=int, default=None, help="first M holdout subjects")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--holdout", type=Path, default=REPO_ROOT / DEFAULT_HOLDOUT_PATH)
    parser.add_argument("--no-bootstrap", action="store_true", help="skip the CI (first cut)")
    args = parser.parse_args(argv)
    partial = args.subjects is not None or args.impostors is not None
    if partial and args.out is None:
        parser.error(
            "--subjects/--impostors make a partial run; pass --out so artifacts/ is untouched"
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = _parse_args(argv)
    started = time.perf_counter()
    out_dir = args.out if args.out is not None else DEFAULT_OUT
    try:
        git_head = check_holdout_committed(args.holdout, REPO_ROOT)
        changes = working_tree_changes(REPO_ROOT)
    except PreconditionError as exc:
        print(f"refusing to run: {exc}", file=sys.stderr)
        return 2
    if changes and out_dir.resolve().is_relative_to((REPO_ROOT / "artifacts").resolve()):
        listed = "\n  ".join(changes[:20])
        print(
            f"refusing to run: artifacts/ must come from committed code; uncommitted:\n  {listed}",
            file=sys.stderr,
        )
        return 2

    record = load_holdout_record(args.holdout)
    full_holdout = frozenset(record.impostor_holdout)
    enrollable = record.enrollable if args.subjects is None else record.enrollable[: args.subjects]
    impostors = (
        record.impostor_holdout
        if args.impostors is None
        else record.impostor_holdout[: args.impostors]
    )
    assert_holdout_excluded(enrollable, full_holdout)

    streaming = StreamingConfig()
    check_margins(streaming, 160.0)
    subjects = sorted((*enrollable, *impostors))
    print(f"extracting bounded-context features: {len(subjects)} subjects x 2 runs", flush=True)
    matrices = [
        process_recording_bounded(recording, streaming)
        for recording in load_baseline_recordings(subjects, args.data_dir, skip_failures=False)
    ]
    print(f"  done in {time.perf_counter() - started:.0f} s", flush=True)

    provenance: dict[str, Any] = {
        "git_head": git_head,
        "git_dirty": bool(changes),
        "holdout_record": {
            "path": args.holdout.resolve().relative_to(REPO_ROOT).as_posix(),
            "selected_at_utc": record.selected_at_utc,
            "seed": record.seed,
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "mne": mne.__version__,
        },
    }
    try:
        summary = run(
            matrices,
            enrollable=tuple(enrollable),
            holdout=full_holdout,
            out_dir=out_dir,
            streaming=streaming,
            bootstrap=not args.no_bootstrap,
            partial=args.subjects is not None or args.impostors is not None,
            provenance=provenance,
        )
    except AssertionError as exc:
        print(f"aborted, no artifacts written: {exc}", file=sys.stderr)
        return 1

    headline = summary["splits"][HEADLINE_SPLIT]
    print(
        f"\nEER {headline['eer']['protected']:.3f}; FRR at FAR {HEADLINE_FAR:.0%} "
        f"{headline['headline_frr_at_far']['frr']:.3f} "
        f"(under-resolved: {headline['headline_frr_at_far']['under_resolved']})"
    )
    print(f"artifacts written to {out_dir} in {time.perf_counter() - started:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
