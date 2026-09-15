"""Cohort-only score and LLR distributions, for choosing session thresholds.

Genuine against cohort-impostor windows, cross-condition, as window-level Hamming similarity
of protected templates (steps of 1/64) and as decision layer v0 LLR, cross-fitted (D-021).
Broken out for the worst decile, ranked on cohort per-subject EER. What is loaded and what is
not, and the dirty-tree rule, are described in scripts/measurements/cohort_scores.py.

Usage:
    python -m scripts.measurements.cohort_score_distributions
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from neuroauth.verification.metrics import (
    FAR_TARGETS,
    ScoreTable,
    eer_value,
    equal_error_rate,
    error_rates,
    frr_at_far,
    genuine_mask,
    impostor_mask,
    n_impostor_pairs,
)
from scripts.measurements.cohort_scores import REPO_ROOT, compute_cohort_scores, provenance_lines
from scripts.train_baseline import PreconditionError

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
    try:
        scores = compute_cohort_scores(args.data_dir)
    except PreconditionError as exc:
        print(f"refusing to run: {exc}", file=sys.stderr)
        return 2
    for line in provenance_lines(scores):
        print(line)
    worst = frozenset(scores.worst_decile)
    block(scores.protected, "SCORE: Hamming similarity (protected), window level", worst, 4)
    block(scores.decision, "LLR: decision layer v0, cross-fitted, window level", worst, 3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
