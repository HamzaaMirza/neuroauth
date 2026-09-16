"""What the pre-registered revoke dwell costs and buys, cohort only.

D-025 revokes only after `revoke_dwell_decisions` consecutive sub-threshold decisions, counts
scored and quality-ok decisions only (flagged ones are skipped, neither advancing nor
resetting a run), and expires a run whose first and last sub-threshold decisions are more than
`revoke_dwell_max_span_s` apart. Challenge has no dwell and is unchanged, so it is not
repeated here.

Three configurations are measured on cohort data, at the pre-registered levels:

- k = 1: revoke on the first sub-threshold decision (the skip rule still applies).
- k = 3, span 8 s: the pre-registered parameter set.
- k = 3, no span: the same dwell without the time bound, so the bound's own effect is
  visible rather than mixed into the k = 1 against k = 3 comparison.

What is measured: genuine-only sessions revoked (the false-revoke rate), self-splice sessions
revoked (the splice control at the dwell), impostor swaps revoked within the horizon with the
timing of the k-th sub-threshold decision, and impostor-only sessions never revoked.

Everything is cohort only; see cohort_scores.py.

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
    Mask,
    decision_delay,
    ema_matrix,
    pair_matrix,
    pair_quality,
    settled_mask,
    swap_sequences,
)
from scripts.train_baseline import PreconditionError

CONFIGURATIONS = (
    ("k=1", 1, float("inf")),
    (
        f"k={PRE_REGISTERED_THRESHOLDS.revoke_dwell_decisions}, "
        f"span {PRE_REGISTERED_THRESHOLDS.revoke_dwell_max_span_s:g}s",
        PRE_REGISTERED_THRESHOLDS.revoke_dwell_decisions,
        PRE_REGISTERED_THRESHOLDS.revoke_dwell_max_span_s,
    ),
    (
        f"k={PRE_REGISTERED_THRESHOLDS.revoke_dwell_decisions}, no span",
        PRE_REGISTERED_THRESHOLDS.revoke_dwell_decisions,
        float("inf"),
    ),
)


def run_completed(
    values: Array, quality: Mask, times: Array, level: float, dwell: int, max_span_s: float
) -> NDArray[np.bool_]:
    """Per decision: does a dwell run complete there?

    Only scored, quality-ok decisions advance or reset a run; flagged ones are skipped
    (D-025). A run whose first and current sub-threshold decisions are more than max_span_s
    apart expires, and the decision that overran it starts a new run.
    """
    completed = np.zeros(values.shape, dtype=np.bool_)
    for row in range(values.shape[0]):
        count = 0
        start = 0.0
        for column in range(values.shape[1]):
            if not quality[row, column]:
                continue
            if values[row, column] >= level:
                count = 0
                continue
            if count and times[column] - start > max_span_s:
                count = 0
            if count == 0:
                start = float(times[column])
            count += 1
            if count >= dwell:
                completed[row, column] = True
    return completed


def revoked_share(completed: NDArray[np.bool_], window: NDArray[np.bool_]) -> float:
    return float(np.mean(completed[:, window].any(axis=1)))


def detection_delays(
    completed: NDArray[np.bool_], confidence: Array, times: Array, level: float
) -> tuple[float, float, float]:
    """(share revoked within the horizon, median, p90) over swaps armed at the swap time."""
    after = times > SWAP_AT_S
    last_before = int(np.flatnonzero(times <= SWAP_AT_S)[-1])
    armed = confidence[:, last_before] >= level
    reached = completed & after
    delay = np.where(reached.any(axis=1), times[reached.argmax(axis=1)] - SWAP_AT_S, np.inf)
    delay = delay[armed]
    delay[delay > DETECTION_HORIZON_S] = np.inf
    if not delay.size:
        return float("nan"), float("nan"), float("nan")
    median, p90 = np.quantile(delay, [0.5, 0.9], method="inverted_cdf")
    return float(np.isfinite(delay).mean()), float(median), float(p90)


def group_mask(
    subjects: NDArray[np.int64], worst: NDArray[np.int64], keep: bool | None
) -> NDArray[np.bool_]:
    """All sessions, or only those whose claimed subject is (or is not) in the worst decile."""
    if keep is None:
        return np.ones(subjects.size, dtype=np.bool_)
    mask: NDArray[np.bool_] = np.isin(subjects, worst) == keep
    return mask


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
    quality = pair_quality(scores.protected)
    times = onsets + decision_delay(scores.streaming)
    genuine_rows = claimed == source
    genuine_subjects, impostor_subjects = claimed[genuine_rows], claimed[~genuine_rows]
    genuine = ema_matrix(values[genuine_rows], times, half_life)
    impostor = ema_matrix(values[~genuine_rows], times, half_life)
    self_subjects, _, self_values, self_quality = self_splice_sequences(scores, args.data_dir)
    self_splice = ema_matrix(self_values, times, half_life)
    swap_tables, swap_quality = swap_sequences(scores, args.data_dir)
    swap_claimed, _, _, swap_values = swap_tables["score"]
    swaps = ema_matrix(swap_values, times, half_life)

    settled = settled_mask(times, half_life)
    horizon = (times > SWAP_AT_S) & (times <= SWAP_AT_S + DETECTION_HORIZON_S)
    flagged = 100.0 * float(1.0 - quality.mean())
    print(
        f"\nREVOKE DWELL at the pre-registered parameters: score, h = {half_life:g} s, revoke "
        f"{level:g}. Flagged decisions ({flagged:.1f}% of windows) are skipped by every run. "
        "genuine/impostor cover settled decisions, self-splice the horizon window; swap "
        f"columns cover swaps armed at {SWAP_AT_S:g} s, timed at the k-th sub-threshold "
        "decision. Challenge is unchanged: it has no dwell."
    )
    header = ["dwell", "group", "genuine%", "selfSplice%", "swap%", "t50", "t90", "escape%"]
    widths = [16, 14, 10, 13, 8, 6, 6, 9]
    print("  " + "".join(f"{name:>{w}}" for name, w in zip(header, widths, strict=True)))
    for label, dwell, span in CONFIGURATIONS:
        completed = {
            "genuine": run_completed(genuine, quality[genuine_rows], times, level, dwell, span),
            "impostor": run_completed(impostor, quality[~genuine_rows], times, level, dwell, span),
            "self": run_completed(self_splice, self_quality, times, level, dwell, span),
            "swap": run_completed(swaps, swap_quality, times, level, dwell, span),
        }
        for group, keep in (("all", None), ("worst decile", True), ("others", False)):
            genuine_share = revoked_share(
                completed["genuine"][group_mask(genuine_subjects, worst, keep)], settled
            )
            self_share = revoked_share(
                completed["self"][group_mask(self_subjects, worst, keep)], horizon
            )
            escape = 1.0 - revoked_share(
                completed["impostor"][group_mask(impostor_subjects, worst, keep)], settled
            )
            swap_rows = group_mask(swap_claimed, worst, keep)
            caught, median, p90 = detection_delays(
                completed["swap"][swap_rows], swaps[swap_rows], times, level
            )
            cells = [
                label,
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
