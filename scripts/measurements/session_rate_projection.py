"""Projecting genuine false-challenge and false-revoke rates to longer sessions, cohort only.

Every measured rate comes from a 55 s replay: 56 decisions, about 51 s of them settled at a
4 s half-life. Real sessions are longer, so the rates for 5, 10, and 30 minutes below are
projections, not measurements.

Method. For each enrolled subject, the genuine-only session's settled confidence is scanned
for downcrossings of the level (a decision below it whose predecessor was not, plus the first
decision if it starts below). That gives a per-subject crossing rate in events per minute.
Two projections follow:

- Per subject, Poisson: P(at least one crossing in T) = 1 - exp(-rate * T). Averaged over a
  group, that is the expected share of sessions affected. Revocation is terminal, so for the
  revoke level this is the share of users revoked at least once in T.
- Pooled, Poisson at the group's overall rate (total crossings over total settled time),
  which describes an average user rather than the mix of users.

Assumptions, all of which can only push the real numbers up:

- Independence between crossings in time. Scores are autocorrelated (about 0.30 at lag 1,
  near zero past lag 5), and crossings cluster, so a Poisson model overstates how evenly
  crossings spread and understates the chance of long quiet stretches.
- Stationarity. A 30-minute session is assumed to look like these 55 s of resting EEG. The
  first-half against second-half rates below are a weak check inside 55 s. Drowsiness,
  movement, and electrode drift over half an hour are not represented, and single-session
  data cannot bound them.
- The same subjects behave the same way for longer. Per-subject rates capture that some
  users never cross and others cross repeatedly; the pooled projection deliberately ignores
  it, and the two together bracket the answer.

Cohort only; see cohort_scores.py.

Usage:
    python -m scripts.measurements.session_rate_projection
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from neuroauth.session.logic import PRE_REGISTERED_THRESHOLDS
from scripts.measurements.cohort_scores import (
    REPO_ROOT,
    compute_cohort_scores,
    provenance_lines,
)
from scripts.measurements.swap_dynamics import (
    Array,
    decision_delay,
    ema_matrix,
    pair_matrix,
    settled_mask,
)
from scripts.train_baseline import PreconditionError

HORIZONS_MIN = (5.0, 10.0, 30.0)


def crossings_per_minute(confidence: Array, duration_min: float, level: float) -> Array:
    """Downcrossings of the level per minute, per session."""
    below = confidence < level
    entries = below[:, 0].astype(np.int64) + (below[:, 1:] & ~below[:, :-1]).sum(axis=1)
    return entries / duration_min


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Project false-cross rates to longer sessions.")
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
    half_life = thresholds.ema_half_life_s
    claimed, source, onsets, values = pair_matrix(scores.protected)
    times = onsets + decision_delay(scores.streaming)
    genuine = claimed == source
    subjects = claimed[genuine]
    settled = settled_mask(times, half_life)
    confidence = ema_matrix(values[genuine], times, half_life)[:, settled]
    settled_times = times[settled]
    duration_min = float(settled_times[-1] - settled_times[0]) / 60.0
    half = confidence.shape[1] // 2
    worst = np.array(sorted(scores.worst_decile), dtype=np.int64)

    print(
        f"\nLONGER-SESSION PROJECTION at the pre-registered parameters: score, "
        f"h = {half_life:g} s. Measured on {confidence.shape[1]} settled decisions per "
        f"session ({60 * duration_min:.0f} s). Poisson projections; see the module docstring "
        "for the assumptions."
    )
    for name, level in (
        ("challenge", thresholds.challenge_below),
        ("revoke", thresholds.revoke_below),
    ):
        for group, keep in (("worst decile", True), ("others", False)):
            mask = np.isin(subjects, worst) == keep
            group_confidence = confidence[mask]
            rate = crossings_per_minute(group_confidence, duration_min, level)
            pooled = float(rate.mean())
            first = crossings_per_minute(group_confidence[:, :half], duration_min / 2, level)
            second = crossings_per_minute(group_confidence[:, half:], duration_min / 2, level)
            print(
                f"\n  {name} at {level:g}, {group}: {mask.sum()} sessions; "
                f"{100 * float(np.mean(rate > 0)):.1f}% cross at least once in "
                f"{60 * duration_min:.0f} s; crossings per minute: pooled {pooled:.3f}, "
                f"median {float(np.median(rate)):.3f}, p90 {float(np.percentile(rate, 90)):.3f}"
            )
            print(
                f"    stationarity check, crossings per minute: first half "
                f"{float(first.mean()):.3f}, second half {float(second.mean()):.3f}"
            )
            print(
                f"    {'horizon':>9}{'per-subject P(>=1)':>20}{'pooled P(>=1)':>15}"
                f"{'expected crossings':>20}"
            )
            for horizon in HORIZONS_MIN:
                per_subject = float(np.mean(1.0 - np.exp(-rate * horizon)))
                print(
                    f"    {horizon:>7.0f}m{100 * per_subject:>19.1f}%"
                    f"{100 * (1.0 - np.exp(-pooled * horizon)):>14.1f}%"
                    f"{pooled * horizon:>20.2f}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
