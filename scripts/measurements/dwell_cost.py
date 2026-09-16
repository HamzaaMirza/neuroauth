"""What the pre-registered revoke dwell costs and buys, cohort only.

D-025 revokes only after `revoke_dwell_decisions` consecutive decisions below `revoke_below`,
and challenges on the first decision below `challenge_below`. This measures, at the
pre-registered parameters and on cohort data only, what the dwell changes before the holdout
session run:

- genuine-only sessions that would be revoked (the false-revoke rate);
- self-splice sessions that would be revoked (the splice control, re-run at the dwell);
- impostor swaps: the share revoked within the horizon, and how much later it happens;
- impostor-only sessions never revoked.

A run of k sub-threshold decisions is what the state machine revokes on, so a revocation is
timed at the k-th decision of the run, not the first.

Everything is cohort only; see cohort_scores.py. The dwell applies to revoke alone, so the
challenge rates are unchanged and are not repeated here.

Usage:
    python -m scripts.measurements.dwell_cost
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from neuroauth.session.logic import PRE_REGISTERED_THRESHOLDS
from neuroauth.session.replay import DETECTION_HORIZON_S, SWAP_AT_S
from scripts.measurements.cohort_scores import (
    REPO_ROOT,
    compute_cohort_scores,
    provenance_lines,
)
from scripts.measurements.self_splice_control import self_splice_sequences
from scripts.measurements.swap_dynamics import (
    Array,
    decision_delay,
    ema_matrix,
    pair_matrix,
    settled_mask,
    swap_sequences,
)
from scripts.train_baseline import PreconditionError

DWELLS = (1, 3)


def run_reached(confidence: Array, level: float, dwell: int) -> NDArray[np.bool_]:
    """Per decision: has confidence been below the level for `dwell` decisions in a row?"""
    below = confidence < level
    runs = np.zeros(below.shape, dtype=np.int64)
    runs[:, 0] = below[:, 0]
    for i in range(1, below.shape[1]):
        runs[:, i] = np.where(below[:, i], runs[:, i - 1] + 1, 0)
    reached: NDArray[np.bool_] = runs >= dwell
    return reached


def revoked_share(confidence: Array, level: float, dwell: int, window: NDArray[np.bool_]) -> float:
    return float(np.mean(run_reached(confidence, level, dwell)[:, window].any(axis=1)))


def group_mask(
    subjects: NDArray[np.int64], worst: NDArray[np.int64], keep: bool | None
) -> NDArray[np.bool_]:
    """All sessions, or only those whose claimed subject is (or is not) in the worst decile."""
    if keep is None:
        return np.ones(subjects.size, dtype=np.bool_)
    mask: NDArray[np.bool_] = np.isin(subjects, worst) == keep
    return mask


def detection_delays(
    confidence: Array, times: Array, level: float, dwell: int
) -> tuple[float, float, float]:
    """(share revoked within the horizon, median, p90) over swaps armed at the swap time."""
    after = times > SWAP_AT_S
    last_before = int(np.flatnonzero(times <= SWAP_AT_S)[-1])
    armed = confidence[:, last_before] >= level
    reached = run_reached(confidence, level, dwell) & after
    delay = np.where(reached.any(axis=1), times[reached.argmax(axis=1)] - SWAP_AT_S, np.inf)
    delay = delay[armed]
    delay[delay > DETECTION_HORIZON_S] = np.inf
    if not delay.size:
        return float("nan"), float("nan"), float("nan")
    median, p90 = np.quantile(delay, [0.5, 0.9], method="inverted_cdf")
    return float(np.isfinite(delay).mean()), float(median), float(p90)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cost of the revoke dwell, cohort only.")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    args = parser.parse_args(argv)
    try:
        scores = compute_cohort_scores(args.data_dir)
    except PreconditionError as exc:
        print(f"refusing to run: {exc}", file=sys.stderr)
        return 2
    for line in provenance_lines(scores):
        print(line)

    thresholds = PRE_REGISTERED_THRESHOLDS
    level = thresholds.revoke_below
    half_life = thresholds.ema_half_life_s
    worst = np.array(sorted(scores.worst_decile), dtype=np.int64)

    claimed, source, onsets, values = pair_matrix(scores.protected)
    times = onsets + decision_delay(scores.streaming)
    genuine_rows = claimed == source
    genuine_subjects = claimed[genuine_rows]
    genuine = ema_matrix(values[genuine_rows], times, half_life)
    impostor_subjects = claimed[~genuine_rows]
    impostor = ema_matrix(values[~genuine_rows], times, half_life)
    self_subjects, _, self_values = self_splice_sequences(scores, args.data_dir)
    self_splice = ema_matrix(self_values, times, half_life)
    swap_claimed, _, _, swap_values = swap_sequences(scores, args.data_dir)["score"]
    swaps = ema_matrix(swap_values, times, half_life)

    settled = settled_mask(times, half_life)
    horizon = (times > SWAP_AT_S) & (times <= SWAP_AT_S + DETECTION_HORIZON_S)

    print(
        f"\nREVOKE DWELL at the pre-registered parameters: score, h = {half_life:g} s, revoke "
        f"{level:g}. genuine/self-splice/impostor are shares of sessions revoked (genuine and "
        f"impostor over settled decisions, self-splice within the horizon window); swap "
        f"columns cover swaps armed at {SWAP_AT_S:g} s, timed at the k-th sub-threshold "
        "decision. Challenge is unchanged: it has no dwell."
    )
    header = ["k", "group", "genuine%", "selfSplice%", "swap%", "t50", "t90", "impostorEscape%"]
    widths = [4, 14, 10, 13, 8, 6, 6, 17]
    print("  " + "".join(f"{name:>{w}}" for name, w in zip(header, widths, strict=True)))
    for dwell in DWELLS:
        for group, keep in (("all", None), ("worst decile", True), ("others", False)):
            genuine_share = revoked_share(
                genuine[group_mask(genuine_subjects, worst, keep)], level, dwell, settled
            )
            self_share = revoked_share(
                self_splice[group_mask(self_subjects, worst, keep)], level, dwell, horizon
            )
            escape = 1.0 - revoked_share(
                impostor[group_mask(impostor_subjects, worst, keep)], level, dwell, settled
            )
            caught, median, p90 = detection_delays(
                swaps[group_mask(swap_claimed, worst, keep)], times, level, dwell
            )
            cells = [
                str(dwell),
                group,
                f"{100 * genuine_share:.1f}",
                f"{100 * self_share:.1f}",
                f"{100 * caught:.1f}",
                f"{median:.0f}",
                f"{p90:.0f}",
                f"{100 * escape:.1f}",
            ]
            print("  " + "".join(f"{c:>{w}}" for c, w in zip(cells, widths, strict=True)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
