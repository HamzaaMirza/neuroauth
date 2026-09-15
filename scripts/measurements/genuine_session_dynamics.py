"""Within-session dynamics of genuine scores, cohort only: autocorrelation and EMA confidence.

Evidence for choosing session parameters (half-life, challenge and revoke levels). This is
not session logic: update_session is the author's. The EMA below is the plainest form, and
its choices are stated so the numbers can be read against whatever the author builds:

- time-based decay: c <- c + alpha * (x - c), with alpha = 1 - 2^(-dt / half_life) and dt the
  gap between decision times (1 s in a replay);
- initialized to the first window's value, with no prior;
- every window averaged, quality-flagged ones included (D-015); no window here is unscorable;
- "settled" decisions are those at least one half-life after the session's first decision.

Genuine-only replays are simulated offline on the per-window scores a replay produces. A
replayed stream's windows are bit-identical to offline bounded-context features
(test_replayed_stream_scores_match_offline_scoring_bit_for_bit), and decision time is onset +
window + right margin. There is one session per enrolled subject: their eyes-closed recording
against their own template, 56 decisions over 55 s.

Autocorrelation is pooled within sessions, with each session's own mean removed, which biases
it down by roughly 1/56. Lag 1 shares half its samples (2 s windows, 1 s hop). Bounded-context
filtering spans 6 s, so lags up to 5 share some filter context; longer lags share neither.

What is loaded and what is not, the cohort-ranked worst decile, and the dirty-tree rule are
described in scripts/measurements/cohort_scores.py.

Usage:
    python -m scripts.measurements.genuine_session_dynamics
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from neuroauth.config import StreamingConfig
from neuroauth.verification.metrics import ScoreTable, genuine_mask
from scripts.measurements.cohort_scores import REPO_ROOT, compute_cohort_scores, provenance_lines
from scripts.train_baseline import PreconditionError

MAX_LAG = 10
HALF_LIVES_S = (2.0, 4.0, 6.0, 8.0, 12.0)
PERCENTILES = (1, 5, 10, 25, 50, 75, 90)

Session = tuple[NDArray[np.float64], NDArray[np.float64]]


def genuine_sessions(table: ScoreTable, streaming: StreamingConfig) -> dict[int, Session]:
    """Claimed subject -> (decision times, genuine window values), in time order."""
    delay = streaming.pipeline.window.window_s + streaming.context.right_margin_s
    genuine = genuine_mask(table)
    sessions: dict[int, Session] = {}
    for subject in np.unique(table.claimed_subject[genuine]).tolist():
        rows = genuine & (table.claimed_subject == subject)
        order = np.argsort(table.probe_onset_s[rows], kind="stable")
        sessions[int(subject)] = (
            table.probe_onset_s[rows][order] + delay,
            table.scores[rows][order],
        )
    return sessions


def pooled_autocorrelation(values: list[NDArray[np.float64]], max_lag: int) -> NDArray[np.float64]:
    centered = [v - v.mean() for v in values]
    denominator = sum(float(c @ c) for c in centered)
    return np.array(
        [
            sum(float(c[:-lag] @ c[lag:]) for c in centered) / denominator
            for lag in range(1, max_lag + 1)
        ]
    )


def per_session_lag1(values: list[NDArray[np.float64]]) -> NDArray[np.float64]:
    out = []
    for v in values:
        c = v - v.mean()
        out.append(float(c[:-1] @ c[1:]) / float(c @ c))
    return np.array(out)


def ema(
    times: NDArray[np.float64], values: NDArray[np.float64], half_life_s: float
) -> NDArray[np.float64]:
    confidence = np.empty_like(values)
    current = float(values[0])
    confidence[0] = current
    for i in range(1, values.size):
        alpha = 1.0 - 0.5 ** ((times[i] - times[i - 1]) / half_life_s)
        current += alpha * (float(values[i]) - current)
        confidence[i] = current
    return confidence


def ema_row(sessions: list[Session], half_life_s: float | None, digits: int, label: str) -> str:
    settled_all: list[NDArray[np.float64]] = []
    session_sd: list[float] = []
    minima_settled: list[float] = []
    minima_all: list[float] = []
    for times, values in sessions:
        conf = values if half_life_s is None else ema(times, values, half_life_s)
        settled = conf if half_life_s is None else conf[times >= times[0] + half_life_s]
        settled_all.append(settled)
        session_sd.append(float(settled.std()))
        minima_settled.append(float(settled.min()))
        minima_all.append(float(conf.min()))
    pooled = np.concatenate(settled_all)
    cells = [f"{v:.{digits}f}" for v in np.percentile(pooled, PERCENTILES)]
    cells.append(f"{np.mean(session_sd):.{digits}f}")
    cells += [
        f"{np.min(minima_settled):.{digits}f}",
        f"{np.percentile(minima_settled, 10):.{digits}f}",
        f"{np.median(minima_settled):.{digits}f}",
        f"{np.median(minima_all):.{digits}f}",
    ]
    width = digits + 5
    return f"  {label:<14}{pooled.size:>7}  " + "".join(f"{c:>{width}}" for c in cells)


def ema_header(digits: int) -> str:
    width = digits + 5
    names = [f"p{p}" for p in PERCENTILES] + ["sessSD", "minMin", "minP10", "minP50", "allMin50"]
    return f"  {'':<14}{'n':>7}  " + "".join(f"{n:>{width}}" for n in names)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Genuine session dynamics, cohort only.")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    args = parser.parse_args(argv)
    try:
        scores = compute_cohort_scores(args.data_dir)
    except PreconditionError as exc:
        print(f"refusing to run: {exc}", file=sys.stderr)
        return 2
    for line in provenance_lines(scores):
        print(line)

    worst = set(scores.worst_decile)
    units = {
        "score": (genuine_sessions(scores.protected, scores.streaming), 4),
        "llr": (genuine_sessions(scores.decision, scores.streaming), 3),
    }
    score_sessions = units["score"][0]
    gaps = np.concatenate([np.diff(times) for times, _ in score_sessions.values()])
    lengths = sorted({times.size for times, _ in score_sessions.values()})
    print(
        f"genuine sessions: {len(score_sessions)}; decisions per session: {lengths}; "
        f"decision spacing min/max: {gaps.min():.2f}/{gaps.max():.2f} s"
    )

    print(f"\nAUTOCORRELATION of genuine window values, pooled within session (lag 1-{MAX_LAG})")
    groups = {
        "all": lambda s: True,
        "worst": lambda s: s in worst,
        "others": lambda s: s not in worst,
    }
    columns: list[tuple[str, NDArray[np.float64]]] = []
    for unit, (sessions, _) in units.items():
        for name, keep in groups.items():
            series = [values for s, (_, values) in sessions.items() if keep(s)]
            columns.append((f"{unit}:{name}", pooled_autocorrelation(series, MAX_LAG)))
    print(f"  {'lag':<5}" + "".join(f"{name:>14}" for name, _ in columns))
    for lag in range(MAX_LAG):
        print(f"  {lag + 1:<5}" + "".join(f"{values[lag]:>14.3f}" for _, values in columns))
    for unit, (sessions, _) in units.items():
        for name in ("worst", "others"):
            lag1 = per_session_lag1([v for s, (_, v) in sessions.items() if groups[name](s)])
            q25, q50, q75 = np.percentile(lag1, [25, 50, 75])
            print(
                f"  per-session lag-1, {unit}:{name}: median {q50:.3f} "
                f"[IQR {q25:.3f} to {q75:.3f}], n={lag1.size}"
            )

    for unit, (sessions, digits) in units.items():
        for name in ("worst", "others"):
            selected = [session for s, session in sessions.items() if groups[name](s)]
            print(
                f"\nEMA CONFIDENCE, {unit}, {name} "
                f"({len(selected)} genuine sessions). Percentiles pool settled decisions; "
                "sessSD = mean within-session SD (settled); min* = session minima (settled); "
                "allMin50 = median session minimum over all decisions"
            )
            print(ema_header(digits))
            print(ema_row(selected, None, digits, "raw windows"))
            for half_life in HALF_LIVES_S:
                print(ema_row(selected, half_life, digits, f"h = {half_life:g} s"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
