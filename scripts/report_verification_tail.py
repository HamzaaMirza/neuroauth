"""Per-subject tail report for the Phase 2 verification run.

Reads committed artifacts only. Nothing is recomputed from EEG:

    artifacts/verification/{split}_protected_impostor_holdout_per_subject_eer.csv
    artifacts/relative_cross_condition_per_class_f1.csv   (Phase 1)
    artifacts/verification/run_summary.json                (the registered verdicts)

What was registered before the run and what was not:

- Registered (D-022). The P1a heavy-tail verdict and the P1b Spearman concordance, with its
  worst-quartile overlap as the descriptive companion. These are echoed from
  run_summary.json, not recomputed.
- Post hoc (D-024). Requested after the results were seen and reported as descriptive only:
  the decile breakdown of the distribution, the overlap between the worst decile and
  Phase 1's zero-F1 subjects with its hypergeometric tail probability, the cross-split
  comparison, and the oracle bound on what per-subject thresholds could change.

Writes per_subject_tail.json and per_subject_tail.png: EERs, counts, and public dataset
subject ids only (hard rule 3). Writing into artifacts/ requires a clean working tree.

Usage:
    python -m scripts.report_verification_tail
"""

import argparse
import csv
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import rc_context
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from scipy.stats import hypergeom, spearmanr

from neuroauth.cohorts import DEFAULT_HOLDOUT_PATH
from neuroauth.verification.metrics import (
    _BASELINE,
    _FONT,
    _GRID,
    _INK_MUTED,
    _INK_PRIMARY,
    _INK_SECONDARY,
    _SERIES,
    _SURFACE,
    summarize_tail,
)
from scripts.train_baseline import PreconditionError, check_holdout_committed, working_tree_changes

REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFICATION = REPO_ROOT / "artifacts" / "verification"
PHASE1_F1 = REPO_ROOT / "artifacts" / "relative_cross_condition_per_class_f1.csv"
HEADLINE_SPLIT = "cross_condition"
OTHER_SPLIT = "temporal"
QUANTILES = (("min", 0), ("p10", 10), ("p25", 25), ("median", 50), ("p75", 75), ("p90", 90))


def per_subject_path(directory: Path, split: str) -> Path:
    return directory / f"{split}_protected_impostor_holdout_per_subject_eer.csv"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def overlap(selected: Sequence[int], marked: set[int], population: int) -> dict[str, Any]:
    """How many of `selected` are `marked`, against a hypergeometric null."""
    hits = sorted(set(selected) & marked)
    return {
        "n_selected": len(selected),
        "n_marked_in_population": len(marked),
        "overlap": len(hits),
        "overlap_subjects": hits,
        "expected_by_chance": len(selected) * len(marked) / population,
        "hypergeometric_p_at_least": float(
            hypergeom.sf(len(hits) - 1, population, len(marked), len(selected))
        ),
    }


def distribution(values: dict[int, float]) -> dict[str, Any]:
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    eers = np.array([eer for _, eer in ordered])
    quantiles = {name: float(np.percentile(eers, q)) for name, q in QUANTILES}
    quantiles["max"] = float(eers.max())
    return {
        "n_subjects": int(eers.size),
        "mean": float(eers.mean()),
        "quantiles": quantiles,
        "decile_means_ascending": [float(chunk.mean()) for chunk in np.array_split(eers, 10)],
        "best_subject": {"subject_id": ordered[0][0], "eer": ordered[0][1]},
        "worst_subject": {"subject_id": ordered[-1][0], "eer": ordered[-1][1]},
    }


def build_report(directory: Path) -> tuple[dict[str, Any], dict[str, dict[int, float]], set[int]]:
    summary = json.loads((directory / "run_summary.json").read_text(encoding="utf-8"))
    splits: dict[str, dict[int, float]] = {}
    counts: dict[str, set[tuple[int, int]]] = {}
    for split in (HEADLINE_SPLIT, OTHER_SPLIT):
        rows = read_rows(per_subject_path(directory, split))
        splits[split] = {int(row["subject_id"]): float(row["eer"]) for row in rows}
        counts[split] = {(int(row["n_genuine"]), int(row["n_impostor"])) for row in rows}
    phase1 = {int(row["subject_id"]): float(row["f1"]) for row in read_rows(PHASE1_F1)}
    zero_f1 = {subject for subject, f1 in phase1.items() if f1 == 0.0}

    headline, other = splits[HEADLINE_SPLIT], splits[OTHER_SPLIT]
    if set(headline) != set(phase1) or set(headline) != set(other):
        raise ValueError("per-subject files and Phase 1 F1 cover different subjects")
    if len(counts[HEADLINE_SPLIT]) != 1:
        raise ValueError("headline subjects have unequal counts; the oracle bound does not hold")
    population = len(headline)
    eers = summary["splits"]
    pooled = eers[HEADLINE_SPLIT]["eer"]["protected"]
    worst = list(summarize_tail(headline, pooled_eer=pooled).worst_decile_subjects)
    other_worst = summarize_tail(other, pooled_eer=eers[OTHER_SPLIT]["eer"]["protected"])
    subjects = sorted(headline)
    rho, _ = spearmanr([headline[s] for s in subjects], [other[s] for s in subjects])
    other_order = sorted(other.items(), key=lambda item: (item[1], item[0]))
    other_rank = {subject: rank for rank, (subject, _) in enumerate(other_order, start=1)}
    registered_p1b = summary["predictions"]["P1b_concordance_with_phase1_f1"]

    report: dict[str, Any] = {
        "what_this_is": (
            "Per-subject EER tail of the Phase 2 headline table (cross-condition, protected, "
            "held-out impostors). Registered verdicts are echoed; everything else is post hoc "
            "and descriptive (D-024). Per-subject EER uses a per-subject threshold, so the tail "
            "exists even with each subject's best possible threshold."
        ),
        "inputs_sha256": {
            path.relative_to(REPO_ROOT).as_posix(): sha256(path)
            for path in (
                per_subject_path(directory, HEADLINE_SPLIT),
                per_subject_path(directory, OTHER_SPLIT),
                PHASE1_F1,
                directory / "run_summary.json",
            )
        },
        "verification_run_git_head": summary["git_head"],
        "registered": {
            "P1a_heavy_tail": summary["predictions"]["P1a_heavy_tail"],
            "P1b_concordance": {
                key: registered_p1b[key]
                for key in (
                    "spearman_rho",
                    "p_value",
                    "zero_f1_in_worst_quartile",
                    "expected_in_worst_quartile_by_chance",
                    "verdict",
                    "caveat",
                )
            },
        },
        "post_hoc": {
            "distribution_cross_condition": {**distribution(headline), "pooled_eer": pooled},
            "distribution_temporal": {
                **distribution(other),
                "genuine_windows_per_subject": sorted({g for g, _ in counts[OTHER_SPLIT]}),
            },
            "worst_decile_cross_condition": {
                "subjects": [
                    {
                        "subject_id": s,
                        "eer": headline[s],
                        "phase1_f1": phase1[s],
                        "temporal_eer": other[s],
                        "temporal_rank_of_89_ascending": other_rank[s],
                    }
                    for s in worst
                ],
                "boundary_note": (
                    "The decile boundary falls inside a tie: the last three members share "
                    "EER 0.3214, and the next subject sits at 0.3208, so edge membership is "
                    "fragile."
                ),
            },
            "worst_decile_vs_phase1_zero_f1": overlap(worst, zero_f1, population),
            "worst_decile_phase1_f1_below_0_1": sum(1 for s in worst if phase1[s] < 0.1),
            "cross_split": {
                "spearman_rho_cross_condition_vs_temporal": float(rho),
                "worst_decile_within_state_eer_below_0_1": sum(1 for s in worst if other[s] < 0.1),
                "worst_deciles_overlap": overlap(
                    worst, set(other_worst.worst_decile_subjects), population
                ),
                "temporal_worst_decile": list(other_worst.worst_decile_subjects),
                "eer_ratio_cross_condition_over_temporal": pooled
                / eers[OTHER_SPLIT]["eer"]["protected"],
                "caveat": (
                    "Temporal per-subject EER rests on 13-14 genuine windows per subject, so "
                    "it is coarse."
                ),
            },
            "oracle_bound_for_per_subject_thresholds": {
                "mean_per_subject_eer": float(np.mean(list(headline.values()))),
                "counts_per_subject": sorted(counts[HEADLINE_SPLIT]),
                "meaning": (
                    "Every headline subject has the same genuine and impostor counts, so with "
                    "each subject at their own EER threshold the pooled FAR and FRR are the "
                    "means of the per-subject values, both at most the mean per-subject EER. "
                    "Those thresholds are chosen on the holdout scores, so this bounds what "
                    "enrollment-calibrated per-subject thresholds could achieve; it is not an "
                    "estimate of it."
                ),
            },
        },
    }
    return report, splits, zero_f1


def _style(axes: Axes) -> None:
    axes.set_facecolor(_SURFACE)
    axes.grid(True, color=_GRID, linewidth=0.5)
    axes.set_axisbelow(True)
    axes.tick_params(colors=_INK_MUTED, labelsize=8, length=0)
    for spine in axes.spines.values():
        spine.set_visible(False)


def _draw(
    axes: Axes, points: list[tuple[float, float, int]], worst: set[int], zero_f1: set[int]
) -> None:
    for x, y, subject in points:
        color = _SERIES[1] if subject in worst else _SERIES[0]
        hollow = subject in zero_f1
        axes.scatter(
            [x],
            [y],
            s=26,
            facecolors=_SURFACE if hollow else color,
            edgecolors=color if hollow else _SURFACE,
            linewidths=1.2 if hollow else 0.8,
            zorder=3,
        )


def _key(label: str, face: str, edge: str) -> Line2D:
    return Line2D(
        [0],
        [0],
        marker="o",
        linestyle="",
        markersize=6,
        markerfacecolor=face,
        markeredgecolor=edge,
        label=label,
    )


def write_figure(
    path: Path,
    splits: dict[str, dict[int, float]],
    worst: set[int],
    zero_f1: set[int],
    pooled: float,
) -> None:
    """Two panels: ranked per-subject EER (cross-condition), and cross-condition vs temporal.

    Color carries one categorical distinction (worst decile or not, palette slots 2 and 1);
    a hollow marker carries Phase 1 F1 = 0, so identity never rests on color alone.
    """
    headline, other = splits[HEADLINE_SPLIT], splits[OTHER_SPLIT]
    ordered = sorted(headline, key=lambda s: (headline[s], s))
    median = float(np.median(list(headline.values())))
    percent = PercentFormatter(1.0, decimals=0)
    label_style = {"color": _INK_SECONDARY, "fontsize": 9}

    with rc_context({"font.family": _FONT}):
        figure = Figure(figsize=(11.0, 5.0), dpi=150, facecolor=_SURFACE)
        FigureCanvasAgg(figure)
        ranked, versus = figure.subplots(1, 2)

        _style(ranked)
        _draw(ranked, [(i + 1, headline[s], s) for i, s in enumerate(ordered)], worst, zero_f1)
        for value, name in ((median, "median"), (pooled, "pooled EER")):
            ranked.axhline(value, color=_BASELINE, linewidth=0.7, zorder=1)
            ranked.text(
                2, value, f" {name} {value:.1%}", color=_INK_SECONDARY, fontsize=7, va="bottom"
            )
        ranked.yaxis.set_major_formatter(percent)
        ranked.set_xlabel("Subjects, ranked by per-subject EER", **label_style)
        ranked.set_ylabel("Per-subject EER, eyes open to eyes closed", **label_style)

        _style(versus)
        top = max(max(headline.values()), max(other.values())) * 1.05
        versus.plot([0, top], [0, top], color=_BASELINE, linewidth=0.7, zorder=1)
        versus.text(top, top, "equal ", color=_INK_MUTED, fontsize=7, ha="right", va="top")
        _draw(versus, [(other[s], headline[s], s) for s in sorted(headline)], worst, zero_f1)
        versus.set_xlim(-0.01, top)
        versus.set_ylim(-0.01, top)
        versus.xaxis.set_major_formatter(percent)
        versus.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        versus.set_xlabel("Per-subject EER, within eyes open (no state change)", **label_style)
        versus.set_ylabel("Per-subject EER, eyes open to eyes closed", **label_style)

        figure.legend(
            handles=[
                _key(f"worst decile ({len(worst)} subjects)", _SERIES[1], _SURFACE),
                _key("other subjects", _SERIES[0], _SURFACE),
                _key("hollow: Phase 1 F1 = 0", _SURFACE, _INK_SECONDARY),
            ],
            loc="upper right",
            ncols=3,
            frameon=False,
            fontsize=8,
            labelcolor=_INK_SECONDARY,
            bbox_to_anchor=(0.985, 0.885),
        )
        figure.subplots_adjust(left=0.065, right=0.985, bottom=0.11, top=0.80, wspace=0.22)
        figure.text(
            0.065,
            0.965,
            "Per-subject EER: the tail behind the pooled number",
            color=_INK_PRIMARY,
            fontsize=11,
            fontweight="bold",
            va="top",
        )
        figure.text(
            0.065,
            0.925,
            "89 enrolled subjects, 20 held-out impostors, protected templates. Each subject at "
            "their own best threshold.",
            color=_INK_SECONDARY,
            fontsize=8,
            va="top",
        )
        figure.savefig(path, facecolor=_SURFACE)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report the per-subject EER tail.")
    parser.add_argument("--artifacts", type=Path, default=VERIFICATION)
    parser.add_argument("--out", type=Path, default=VERIFICATION)
    args = parser.parse_args(argv)
    try:
        git_head = check_holdout_committed(REPO_ROOT / DEFAULT_HOLDOUT_PATH, REPO_ROOT)
        changes = working_tree_changes(REPO_ROOT)
    except PreconditionError as exc:
        print(f"refusing to run: {exc}", file=sys.stderr)
        return 2
    if changes and args.out.resolve().is_relative_to((REPO_ROOT / "artifacts").resolve()):
        print("refusing to run: artifacts/ must come from committed code", file=sys.stderr)
        return 2

    report, splits, zero_f1 = build_report(args.artifacts)
    report["git_head"] = git_head
    report["git_dirty"] = bool(changes)
    args.out.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2) + "\n"
    (args.out / "per_subject_tail.json").write_text(text, encoding="utf-8")
    post_hoc = report["post_hoc"]
    worst = {row["subject_id"] for row in post_hoc["worst_decile_cross_condition"]["subjects"]}
    pooled = post_hoc["distribution_cross_condition"]["pooled_eer"]
    write_figure(args.out / "per_subject_tail.png", splits, worst, zero_f1, pooled)
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
