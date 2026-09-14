"""Run the Phase 1 identification baseline and write the committed artifacts.

Four evaluations, because one number would not be interpretable:

    1. relative     x cross-condition   <- the headline
    2. relative     x temporal
    3. absolute_log x cross-condition   <- the confound comparison (D-004)
    4. absolute_log x temporal

plus one deliberately leaky random split, written to its own artifact and labelled as
a demonstration, to measure what the guarded splits prevent (D-017).

Preconditions, checked before any EEG is loaded:
    - config/impostor_holdout.json is committed and unmodified, so no result can
      predate the holdout commitment (D-008). The list is loaded from that file,
      never re-derived from the seed.
    - No subject about to be used is in the holdout (assert_holdout_excluded).

Every model gets the shuffled-label control. If any control exceeds 3x chance the
script aborts before writing a single artifact. The control catches label leakage
outside the train association; window-overlap leakage is ruled out structurally by
assert_no_window_overlap (D-007). Thresholds, seeds, and split parameters were fixed
before the first real-data run (D-016).

Windows flagged by the quality mask are scored, not excluded; the count is reported
next to each score, and per-recording rates go to quality_flag_rates.csv (D-015).

Band and channel importance are written for every model, with the frontal EOG share
against its uniform baseline (D-004b). The cross-condition models are the
eyes-open-trained ones.

Artifacts hold scores, counts, importances, and quality rates only -- never a feature
vector (hard rule 3).

Usage:
    python -m scripts.train_baseline
    python -m scripts.train_baseline --subjects 5 --out artifacts/smoke   # partial run
"""

import argparse
import json
import platform
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mne
import numpy as np
import sklearn

from neuroauth.cohorts import DEFAULT_HOLDOUT_PATH, assert_holdout_excluded, load_holdout_record
from neuroauth.config import FRONTAL_EOG_CHANNELS, FeatureConfig, Normalization, PipelineConfig
from neuroauth.dsp.io import load_baseline_recordings
from neuroauth.dsp.pipeline import process_recording
from neuroauth.dsp.types import FeatureMatrix
from neuroauth.models.baseline import (
    BaselineConfig,
    band_importance,
    channel_importance,
    importance_share,
    predict_baseline,
    run_shuffled_label_control,
    train_baseline,
)
from neuroauth.models.evaluation import (
    CONTROL_MAX_CHANCE_RATIO,
    MATERIAL_DELTA,
    IdentificationReport,
    check_shuffled_label_control,
    compare_normalizations,
    evaluate_identification,
    summarize_quality,
    write_quality_report,
    write_report,
)
from neuroauth.models.splits import (
    concatenate,
    count_overlapping_test_windows,
    cross_condition_split,
    leaky_random_split,
    temporal_split,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

NORMALIZATIONS: tuple[Normalization, ...] = ("relative", "absolute_log")
SPLIT_KINDS: tuple[str, ...] = ("cross_condition", "temporal")
HEADLINE: tuple[Normalization, str] = ("relative", "cross_condition")
TRAIN_CONDITION = "eyes_open"
TEST_CONDITION = "eyes_closed"
TRAIN_FRACTION = 0.7
CONTROL_SEED = 20260914
LEAKY_SPLIT_SEED = 20260915
LEAKY_SPLIT_KIND = "leaky_random_demonstration"
LEAKAGE_ARTIFACT = "leakage_demonstration.json"


class PreconditionError(RuntimeError):
    """A condition that must hold before any EEG is loaded does not."""


@dataclass(frozen=True)
class RunResult:
    """One trained-and-scored model, before anything is written."""

    report: IdentificationReport
    trained_on: str
    band_importance: dict[str, float]
    channel_importance: dict[str, float]
    frontal_share: float
    uniform_share: float


def check_holdout_committed(holdout_path: Path, repo_root: Path) -> str:
    """Refuse to run unless the holdout file is in HEAD and unmodified.

    Turns "no result before the holdout commitment" into a checked precondition
    rather than a habit (D-008). A file that is only staged does not pass. This
    cannot check that the commit was pushed.

    Args:
        holdout_path: The holdout record.
        repo_root: The git working tree it must be committed in.

    Returns:
        The HEAD commit sha, recorded in run_summary.json.

    Raises:
        PreconditionError: If git is unavailable, the file is outside the repository,
            absent from HEAD, or differs from its committed version.
    """
    try:
        relative = holdout_path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        raise PreconditionError(f"{holdout_path} is outside the repository") from None

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                ["git", *args], cwd=repo_root, capture_output=True, text=True, check=False
            )
        except FileNotFoundError:
            raise PreconditionError("git is not available to verify the holdout commit") from None

    if git("cat-file", "-e", f"HEAD:{relative}").returncode != 0:
        raise PreconditionError(
            f"{relative} is not committed. Commit it on its own, before any results."
        )
    if git("diff", "--quiet", "HEAD", "--", relative).returncode != 0:
        raise PreconditionError(f"{relative} differs from its committed version")
    head = git("rev-parse", "HEAD")
    if head.returncode != 0:
        raise PreconditionError("could not read the HEAD commit")
    return head.stdout.strip()


def load_feature_sets(
    subjects: Sequence[int],
    data_dir: Path,
    configs: dict[Normalization, PipelineConfig],
) -> dict[Normalization, list[FeatureMatrix]]:
    """Load each recording once and extract features under every config.

    A recording that fails to load aborts the run: every baseline recording passed
    the structural check (D-014), so a failure now is a problem to fix, not a
    subject to drop silently.
    """
    sets: dict[Normalization, list[FeatureMatrix]] = {name: [] for name in configs}
    for recording in load_baseline_recordings(subjects, data_dir, skip_failures=False):
        for normalization, config in configs.items():
            sets[normalization].append(process_recording(recording, config))
    return sets


def evaluate_split(
    matrices: list[FeatureMatrix],
    split_kind: str,
    *,
    normalization: Normalization,
    fingerprint: str,
    window_s: float,
    n_holdout: int,
    baseline: BaselineConfig,
) -> RunResult:
    """Train, score, run the shuffled-label control, and gate on it."""
    x, y, names = concatenate(matrices)
    window_ok = np.concatenate([matrix.quality.window_ok for matrix in matrices])
    labels = tuple(int(subject) for subject in np.unique(y))

    if split_kind == "cross_condition":
        train, test = cross_condition_split(matrices, TRAIN_CONDITION, TEST_CONDITION)
        trained_on = TRAIN_CONDITION
    elif split_kind == "temporal":
        train, test = temporal_split(matrices, TRAIN_FRACTION, window_s=window_s, guard_s=window_s)
        trained_on = f"{TRAIN_CONDITION}+{TEST_CONDITION}"
    else:
        raise ValueError(f"unknown split kind {split_kind!r}")

    model = train_baseline(x[train], y[train], baseline)
    predictions, _ = predict_baseline(model, x[test])
    control = run_shuffled_label_control(
        x[train], y[train], x[test], y[test], labels, baseline, seed=CONTROL_SEED
    )
    report = evaluate_identification(
        y[test],
        predictions,
        labels,
        split_kind=split_kind,
        normalization=normalization,
        config_fingerprint=fingerprint,
        n_train=int(train.size),
        n_test_not_ok=int((~window_ok[test]).sum()),
        n_subjects_enrollable=len(labels),
        n_subjects_impostor_holdout=n_holdout,
        shuffled_label_macro_f1=control,
    )
    check_shuffled_label_control(report)

    channels = channel_importance(model, names)
    share, uniform = importance_share(channels, FRONTAL_EOG_CHANNELS)
    return RunResult(
        report=report,
        trained_on=trained_on,
        band_importance=band_importance(model, names),
        channel_importance=channels,
        frontal_share=share,
        uniform_share=uniform,
    )


def run_leakage_demonstration(
    matrices: list[FeatureMatrix],
    honest: IdentificationReport,
    *,
    fingerprint: str,
    window_s: float,
    n_holdout: int,
    baseline: BaselineConfig,
) -> tuple[IdentificationReport, dict[str, Any]]:
    """Score a deliberately leaky random split against the guarded temporal split.

    The comparison is against the temporal split, not cross-condition: both split
    windows within the same recordings, so the only difference is whether windows
    that share samples can straddle the boundary (D-017).
    """
    x, y, _ = concatenate(matrices)
    window_ok = np.concatenate([matrix.quality.window_ok for matrix in matrices])
    labels = tuple(int(subject) for subject in np.unique(y))
    train, test = leaky_random_split(matrices, TRAIN_FRACTION, seed=LEAKY_SPLIT_SEED)

    model = train_baseline(x[train], y[train], baseline)
    predictions, _ = predict_baseline(model, x[test])
    control = run_shuffled_label_control(
        x[train], y[train], x[test], y[test], labels, baseline, seed=CONTROL_SEED
    )
    leaky = evaluate_identification(
        y[test],
        predictions,
        labels,
        split_kind=LEAKY_SPLIT_KIND,
        normalization=honest.normalization,
        config_fingerprint=fingerprint,
        n_train=int(train.size),
        n_test_not_ok=int((~window_ok[test]).sum()),
        n_subjects_enrollable=len(labels),
        n_subjects_impostor_holdout=n_holdout,
        shuffled_label_macro_f1=control,
    )
    check_shuffled_label_control(leaky)

    overlapping = count_overlapping_test_windows(matrices, train, test, window_s)
    fraction = overlapping / test.size
    inflation = leaky.macro_f1 - honest.macro_f1
    artifact = {
        "label": "DELIBERATELY LEAKY SPLIT -- demonstration only, not a result",
        "what_this_is": (
            "A shuffled row split over 50%-overlapping windows, run to measure what the "
            "guarded splits prevent. Its test windows share samples with training windows "
            "by construction. See docs/DECISIONS.md D-017."
        ),
        "normalization": honest.normalization,
        "honest_split": f"{honest.split_kind}, guard {window_s:g} s, no shared samples",
        "honest_macro_f1": honest.macro_f1,
        "leaky_macro_f1": leaky.macro_f1,
        "inflation_leaky_minus_honest": inflation,
        "chance_level": leaky.chance_level,
        "leaky_shuffled_label_macro_f1": control,
        "leaky_n_train": int(train.size),
        "leaky_n_test": int(test.size),
        "leaky_test_windows_sharing_samples_with_train": overlapping,
        "leaky_fraction_sharing_samples": fraction,
        "train_fraction": TRAIN_FRACTION,
        "seed": LEAKY_SPLIT_SEED,
        "config_fingerprint": fingerprint,
        "interpretation": (
            f"A random split over overlapping windows scores {leaky.macro_f1:.3f} macro-F1, "
            f"against {honest.macro_f1:.3f} for the guarded temporal split on the same "
            f"recordings: an inflation of {inflation:.3f}. {100 * fraction:.0f}% of the "
            "random split's test windows share samples with a training window. The "
            f"shuffled-label control on the random split reads {control:.3f} "
            f"(chance {leaky.chance_level:.3f})."
        ),
    }
    return leaky, artifact


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _importance_entry(result: RunResult) -> dict[str, Any]:
    return {
        "trained_on": result.trained_on,
        "band_importance": result.band_importance,
        "frontal_eog_channels": list(FRONTAL_EOG_CHANNELS),
        "frontal_importance_share": result.frontal_share,
        "frontal_uniform_share": result.uniform_share,
        "frontal_share_vs_uniform": result.frontal_share / result.uniform_share,
        "channel_importance": result.channel_importance,
    }


def run(
    matrices_by_normalization: dict[Normalization, list[FeatureMatrix]],
    *,
    holdout: frozenset[int],
    out_dir: Path,
    fingerprints: dict[Normalization, str],
    window_s: float,
    baseline: BaselineConfig,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate every model, gate every control, and only then write artifacts.

    Args:
        matrices_by_normalization: One list of per-recording matrices per
            normalization, built from the same recordings.
        holdout: The impostor holdout, as loaded from the committed record.
        out_dir: Artifact directory.
        fingerprints: PipelineConfig.fingerprint() per normalization.
        window_s: Window length; also the temporal-split guard.
        baseline: Random Forest hyperparameters.
        provenance: Merged into run_summary.json (git HEAD, holdout record, versions).

    Returns:
        The run summary, as written to run_summary.json.

    Raises:
        AssertionError: If a holdout subject is present or any shuffled-label control
            fails. Nothing is written in either case.
    """
    for matrices in matrices_by_normalization.values():
        assert_holdout_excluded(
            (matrix.subject_id for matrix in matrices if matrix.subject_id is not None), holdout
        )

    results: dict[tuple[Normalization, str], RunResult] = {}
    for normalization in NORMALIZATIONS:
        for split_kind in SPLIT_KINDS:
            print(f"  training {normalization} x {split_kind}", flush=True)
            results[(normalization, split_kind)] = evaluate_split(
                matrices_by_normalization[normalization],
                split_kind,
                normalization=normalization,
                fingerprint=fingerprints[normalization],
                window_s=window_s,
                n_holdout=len(holdout),
                baseline=baseline,
            )

    print("  training the deliberately leaky random split (demonstration)", flush=True)
    relative = matrices_by_normalization["relative"]
    _, leakage = run_leakage_demonstration(
        relative,
        results[("relative", "temporal")].report,
        fingerprint=fingerprints["relative"],
        window_s=window_s,
        n_holdout=len(holdout),
        baseline=baseline,
    )

    # Every control has passed. Only now does anything touch disk.
    out_dir.mkdir(parents=True, exist_ok=True)
    for (normalization, split_kind), result in results.items():
        write_report(result.report, out_dir, prefix=f"{normalization}_{split_kind}")
    _write_json(
        out_dir / "normalization_comparison.json",
        {
            split_kind: compare_normalizations(
                results[("relative", split_kind)].report,
                results[("absolute_log", split_kind)].report,
            )
            for split_kind in SPLIT_KINDS
        },
    )
    _write_json(
        out_dir / "importance.json",
        {f"{n}_{s}": _importance_entry(result) for (n, s), result in results.items()},
    )
    write_quality_report(summarize_quality(relative), out_dir)
    _write_json(out_dir / LEAKAGE_ARTIFACT, leakage)

    headline = results[HEADLINE].report
    summary: dict[str, Any] = {
        **provenance,
        "headline": {
            "normalization": headline.normalization,
            "split_kind": headline.split_kind,
            "macro_f1": headline.macro_f1,
            "chance_level": headline.chance_level,
            "shuffled_label_macro_f1": headline.shuffled_label_macro_f1,
        },
        "runs": [
            {
                "normalization": normalization,
                "split_kind": split_kind,
                "trained_on": result.trained_on,
                "macro_f1": result.report.macro_f1,
                "chance_level": result.report.chance_level,
                "shuffled_label_macro_f1": result.report.shuffled_label_macro_f1,
                "n_train": result.report.n_train,
                "n_test": result.report.n_test,
                "n_test_not_ok": result.report.n_test_not_ok,
                "frontal_importance_share": result.frontal_share,
                "config_fingerprint": result.report.config_fingerprint,
            }
            for (normalization, split_kind), result in results.items()
        ],
        "leakage_demonstration": {
            key: leakage[key]
            for key in (
                "label",
                "honest_macro_f1",
                "leaky_macro_f1",
                "inflation_leaky_minus_honest",
                "leaky_shuffled_label_macro_f1",
                "leaky_fraction_sharing_samples",
            )
        },
        "n_subjects_evaluated": len(headline.labels),
        "n_subjects_impostor_holdout": len(holdout),
        "config_fingerprints": dict(fingerprints),
        "baseline_config": asdict(baseline),
        "split_parameters": {
            "train_condition": TRAIN_CONDITION,
            "test_condition": TEST_CONDITION,
            "train_fraction": TRAIN_FRACTION,
            "window_s": window_s,
            "guard_s": window_s,
            "control_seed": CONTROL_SEED,
            "leaky_split_seed": LEAKY_SPLIT_SEED,
        },
        "a_priori_thresholds": {
            "control_max_chance_ratio": CONTROL_MAX_CHANCE_RATIO,
            "material_delta": MATERIAL_DELTA,
        },
    }
    _write_json(out_dir / "run_summary.json", summary)
    return summary


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 1 identification baseline.")
    parser.add_argument(
        "--subjects",
        type=int,
        default=None,
        help="use only the first N enrollable subjects (a partial run; requires --out)",
    )
    parser.add_argument("--out", type=Path, default=None, help="artifact directory")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--holdout", type=Path, default=REPO_ROOT / DEFAULT_HOLDOUT_PATH)
    args = parser.parse_args(argv)
    if args.subjects is not None:
        if args.out is None:
            parser.error("--subjects makes a partial run; pass --out so artifacts/ is untouched")
        if args.subjects < 2:
            parser.error("--subjects must be at least 2")
    return args


def _print_summary(summary: dict[str, Any], out_dir: Path, elapsed_s: float) -> None:
    uniform = len(FRONTAL_EOG_CHANNELS) / 64
    print()
    print(
        f"{'normalization':<14}{'split':<17}{'macro-F1':>9}{'shuffled':>10}{'chance':>8}"
        f"{'frontal share':>15}"
    )
    for entry in summary["runs"]:
        print(
            f"{entry['normalization']:<14}{entry['split_kind']:<17}"
            f"{entry['macro_f1']:>9.3f}{entry['shuffled_label_macro_f1']:>10.3f}"
            f"{entry['chance_level']:>8.3f}{entry['frontal_importance_share']:>15.3f}"
        )
    print(f"(frontal share under uniform importance: {uniform:.4f})")
    leak = summary["leakage_demonstration"]
    print(
        f"\nleakage demonstration, relative: guarded temporal {leak['honest_macro_f1']:.3f}, "
        f"deliberately leaky random {leak['leaky_macro_f1']:.3f}, "
        f"shuffled control on leaky {leak['leaky_shuffled_label_macro_f1']:.3f}"
    )
    print(f"\nartifacts written to {out_dir} in {elapsed_s:.0f} s")


def working_tree_changes(repo_root: Path) -> list[str]:
    """Paths with uncommitted changes, tracked or untracked, outside artifacts/.

    Committed artifacts must trace back to committed code: a git_head recorded from a
    dirty tree names a commit that did not produce the numbers. Regenerated artifacts
    themselves are ignored, so a re-run on clean code is not blocked by its own output.

    Args:
        repo_root: The git working tree.

    Returns:
        Changed paths, as git reports them. Empty for a clean tree.

    Raises:
        PreconditionError: If git is unavailable or the status cannot be read.
    """
    try:
        status = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--untracked-files=all",
                "--",
                ".",
                ":(exclude)artifacts",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        raise PreconditionError("git is not available to check the working tree") from None
    if status.returncode != 0:
        raise PreconditionError(f"git status failed: {status.stderr.strip()}")
    return [line[3:] for line in status.stdout.splitlines() if line.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = _parse_args(argv)
    started = time.perf_counter()
    out_dir = args.out if args.out is not None else REPO_ROOT / "artifacts"
    try:
        git_head = check_holdout_committed(args.holdout, REPO_ROOT)
        changes = working_tree_changes(REPO_ROOT)
    except PreconditionError as exc:
        print(f"refusing to run: {exc}", file=sys.stderr)
        return 2
    if changes and out_dir.resolve() == (REPO_ROOT / "artifacts").resolve():
        listed = "\n  ".join(changes[:20])
        print(
            "refusing to run: artifacts/ must come from committed code, and these paths have "
            f"uncommitted changes:\n  {listed}",
            file=sys.stderr,
        )
        return 2

    record = load_holdout_record(args.holdout)
    subjects = record.enrollable if args.subjects is None else record.enrollable[: args.subjects]
    holdout = frozenset(record.impostor_holdout)
    assert_holdout_excluded(subjects, holdout)

    configs = {
        name: PipelineConfig(features=FeatureConfig(normalization=name)) for name in NORMALIZATIONS
    }
    print(f"extracting features: {len(subjects)} enrollable subjects x 2 baseline runs", flush=True)
    matrices = load_feature_sets(subjects, args.data_dir, configs)
    print(f"  done in {time.perf_counter() - started:.0f} s", flush=True)

    provenance: dict[str, Any] = {
        "git_head": git_head,
        "git_dirty": bool(changes),
        "holdout_record": {
            "path": args.holdout.resolve().relative_to(REPO_ROOT).as_posix(),
            "selected_at_utc": record.selected_at_utc,
            "seed": record.seed,
            "n_impostors": record.n_impostors,
        },
        "partial_run": args.subjects is not None,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "mne": mne.__version__,
        },
    }
    try:
        summary = run(
            matrices,
            holdout=holdout,
            out_dir=out_dir,
            fingerprints={name: config.fingerprint() for name, config in configs.items()},
            window_s=configs["relative"].window.window_s,
            baseline=BaselineConfig(),
            provenance=provenance,
        )
    except AssertionError as exc:
        print(f"aborted, no artifacts written: {exc}", file=sys.stderr)
        return 1

    _print_summary(summary, out_dir, time.perf_counter() - started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
