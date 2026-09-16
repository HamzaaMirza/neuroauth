"""EMA confidence for cohort impostors and cohort-only mid-session swaps, and time to cross levels.

Evidence for choosing session parameters. This is not session logic: update_session is the
author's. The EMA is the one in genuine_session_dynamics.py (time-based decay, initialized to
the first window, every window averaged, "settled" = at least one half-life after the first
decision).

Everything is cohort only: claimed and impostor subjects are both enrollable and in the same
fold, never a holdout subject (see cohort_scores.py for what is loaded and asserted).

- Impostor sessions: each enrolled subject's template against every other enrolled subject of
  its fold, using the impostor's eyes-closed recording, 56 decisions each.
- Swaps: for every such (claimed, impostor) pair, the claimed subject's eyes-closed recording
  is spliced at SWAP_AT_S = 30 s onto the impostor's eyes-closed recording, read from the same
  offset, with the registered 0.5 s raw crossfade (session/replay.py). A replay of that stream
  produces, bit for bit:
    * before the splice reaches a window's raw context: the claimed subject's genuine windows;
    * after it has passed: the impostor's windows scored against the claimed template;
    * in between: the windows whose 6 s raw context straddles the splice, which are
      featurized here from the spliced signal.
  Each swap's sequence is assembled from those three parts. One swap per fold is also
  replayed in full and compared.

Timing, precisely, for a swap at 30.0 s (1 s hop, 2 s window, 2 s margins, 0.5 s crossfade):

- decision at 31 s (+1 s): first raw context containing post-swap samples, in the right
  margin only; they reach the window through the filter tail;
- decision at 33 s (+3 s): first window containing post-swap samples;
- decision at 35 s (+5 s): first window entirely after the crossfade, while its left margin
  still holds 1.5 s of pre-swap and crossfade signal;
- decision at 37 s (+7 s): first decision whose whole raw context is impostor signal.

Crossing. A level is crossed at the first decision after the swap where confidence < level.
A swap whose confidence is already below the level at the last decision before the swap is
"not armed": it is counted separately and excluded from timing, so a genuine dip is never
scored as a fast detection. Genuine false-cross is the share of genuine-only sessions whose
settled confidence ever drops below the level. Impostor never-caught is the share of
impostor-only sessions whose settled confidence never drops below it.

The levels are a descriptive grid spanning the observed distributions, not recommendations.

Usage:
    python -m scripts.measurements.swap_dynamics
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from neuroauth.config import StreamingConfig
from neuroauth.dsp.io import load_baseline_recordings
from neuroauth.dsp.streaming import context_window_starts, featurize_windows
from neuroauth.dsp.types import FeatureMatrix
from neuroauth.session.replay import CROSSFADE_S, DETECTION_HORIZON_S, SWAP_AT_S, splice
from neuroauth.templates.cancelable import (
    derive_key,
    hamming_similarity,
    projection_matrix,
    protect,
)
from neuroauth.verification.decision import DecisionLayer, decision_llr
from neuroauth.verification.embedding import embed_values
from neuroauth.verification.metrics import ScoreTable
from neuroauth.verification.protocol import FoldScores
from scripts.evaluate_verification import EVALUATION_MASTER_SECRET
from scripts.measurements.cohort_scores import (
    REPO_ROOT,
    CohortScores,
    compute_cohort_scores,
    provenance_lines,
)
from scripts.train_baseline import PreconditionError

Array = NDArray[np.float64]
Mask = NDArray[np.bool_]
Pairs = tuple[NDArray[np.int64], NDArray[np.int64], Array, Array]

HALF_LIVES_S: tuple[float | None, ...] = (None, 2.0, 4.0, 6.0, 8.0, 12.0)
LEVELS = {
    "score": (0.54, 0.56, 0.58, 0.60, 0.62, 0.64),
    "llr": (-1.0, -0.5, 0.0, 0.5, 1.0, 1.5),
}
DIGITS = {"score": 3, "llr": 2}
TRAJECTORY_OFFSETS_S = (-4, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25)
PERCENTILES = (1, 10, 50, 90, 99)


def h_label(half_life: float | None) -> str:
    return "raw" if half_life is None else f"h={half_life:g}s"


def decision_delay(streaming: StreamingConfig) -> float:
    return streaming.pipeline.window.window_s + streaming.context.right_margin_s


def pair_matrix(table: ScoreTable) -> Pairs:
    """(claimed, source, onsets, values): one row of values per (claimed, source) pair."""
    order = np.lexsort((table.probe_onset_s, table.source_subject, table.claimed_subject))
    n_windows = int(np.unique(table.probe_onset_s).size)
    if order.size % n_windows:
        raise ValueError("pairs do not all have the same number of windows")
    shape = (order.size // n_windows, n_windows)
    onsets = table.probe_onset_s[order].reshape(shape)
    if not np.all(onsets == onsets[0]):
        raise ValueError("pairs have different window onsets")
    return (
        table.claimed_subject[order].reshape(shape)[:, 0],
        table.source_subject[order].reshape(shape)[:, 0],
        onsets[0],
        table.scores[order].reshape(shape),
    )


def pair_quality(table: ScoreTable) -> Mask:
    """probe_window_ok in the same (pair, window) layout as pair_matrix."""
    order = np.lexsort((table.probe_onset_s, table.source_subject, table.claimed_subject))
    n_windows = int(np.unique(table.probe_onset_s).size)
    quality: Mask = table.probe_window_ok[order].reshape(order.size // n_windows, n_windows)
    return quality


def ema_matrix(values: Array, times: Array, half_life: float | None) -> Array:
    """EMA along axis 1. None returns the raw windows."""
    if half_life is None:
        return values
    out = np.empty_like(values)
    current = values[:, 0].copy()
    out[:, 0] = current
    for i in range(1, values.shape[1]):
        alpha = 1.0 - 0.5 ** ((times[i] - times[i - 1]) / half_life)
        current = current + alpha * (values[:, i] - current)
        out[:, i] = current
    return out


def settled_mask(times: Array, half_life: float | None) -> Mask:
    if half_life is None:
        return np.ones_like(times, dtype=np.bool_)
    mask: Mask = times >= times[0] + half_life
    return mask


def splice_parts(
    onsets: Array, sfreq: float, streaming: StreamingConfig
) -> tuple[Mask, Mask, NDArray[np.int64]]:
    """(before, straddling, window starts): which windows' raw context misses the splice."""
    window = round(streaming.pipeline.window.window_s * sfreq)
    left = round(streaming.context.left_margin_s * sfreq)
    right = round(streaming.context.right_margin_s * sfreq)
    swap = round(SWAP_AT_S * sfreq)
    fade = round(CROSSFADE_S * sfreq)
    starts = np.round(onsets * sfreq).astype(np.int64)
    before = starts + window + right <= swap
    after = starts - left >= swap + fade
    return before, ~(before | after), starts


def score_features(
    fold: FoldScores,
    layer: DecisionLayer,
    claimed: int,
    features: FeatureMatrix,
    projection: Array,
) -> tuple[Array, Array]:
    """Protected similarity and LLR against the claimed template, computed as score_fold does."""
    template = fold.templates[claimed]
    embedded = embed_values(fold.model, features.values)
    similarity = hamming_similarity(protect(embedded, projection), template.bits)
    n_rows = similarity.size
    llr = decision_llr(
        layer,
        similarity,
        np.full(n_rows, template.enrollment_score_mean),
        np.full(n_rows, template.enrollment_score_std),
        features.quality.window_ok,
    )
    return similarity, llr


def swap_sequences(scores: CohortScores, data_dir: Path) -> tuple[dict[str, Pairs], Mask]:
    """Per unit: (claimed, impostor, decision times, values) for every cohort swap, and the
    window quality of those same sessions."""
    streaming = scores.streaming
    units = {"score": pair_matrix(scores.protected), "llr": pair_matrix(scores.decision)}
    claimed, source, onsets, _ = units["score"]
    if not (np.array_equal(units["llr"][0], claimed) and np.array_equal(units["llr"][1], source)):
        raise ValueError("score and llr tables disagree on pair order")
    row_of = {(int(c), int(s)): i for i, (c, s) in enumerate(zip(claimed, source, strict=True))}
    times = onsets + decision_delay(streaming)
    straddling: dict[tuple[int, int], tuple[Array, Array]] = {}
    straddling_ok: dict[tuple[int, int], Mask] = {}
    before = np.zeros_like(onsets, dtype=np.bool_)
    mixed = np.zeros_like(onsets, dtype=np.bool_)

    def assemble(unit: str, subject: int, impostor: int) -> Array:
        values = units[unit][3]
        genuine_row = values[row_of[(subject, subject)]]
        sequence = np.where(before, genuine_row, values[row_of[(subject, impostor)]])
        sequence[mixed] = straddling[(subject, impostor)][0 if unit == "score" else 1]
        return sequence

    for fold in scores.folds:
        enrolled = sorted(fold.templates)
        members = sorted({int(s) for s in source[np.isin(claimed, enrolled)]})
        recordings = {
            r.subject_id: r
            for r in load_baseline_recordings(members, data_dir, runs=(2,), skip_failures=False)
        }
        if set(recordings) & scores.holdout:
            raise AssertionError("a holdout subject's recording was loaded")
        sample = recordings[enrolled[0]]
        before, mixed, starts = splice_parts(onsets, sample.sfreq, streaming)
        layer = scores.decision_layers[fold.fold_index]
        checked = False
        for subject in enrolled:
            template = fold.templates[subject]
            key = derive_key(EVALUATION_MASTER_SECRET, template.subject_ref, template.key_version)
            projection = projection_matrix(key, fold.model.n_components, fold.model.n_components)
            for impostor in members:
                if impostor == subject:
                    continue
                spliced = splice(
                    recordings[subject],
                    recordings[impostor],
                    swap_at_s=SWAP_AT_S,
                    crossfade_s=CROSSFADE_S,
                )
                features = featurize_windows(
                    spliced, starts[mixed], sample.sfreq, sample.ch_names, streaming
                )
                straddling[(subject, impostor)] = score_features(
                    fold, layer, subject, features, projection
                )
                straddling_ok[(subject, impostor)] = features.quality.window_ok
                if checked:
                    continue
                checked = True
                full_starts = context_window_starts(spliced.shape[1], sample.sfreq, streaming)
                if not np.array_equal(full_starts, starts):
                    raise ValueError("a full replay's windows differ from the table's windows")
                full = featurize_windows(
                    spliced, full_starts, sample.sfreq, sample.ch_names, streaming
                )
                full_score, full_llr = score_features(fold, layer, subject, full, projection)
                score_diff = np.abs(full_score - assemble("score", subject, impostor))
                llr_diff = np.abs(full_llr - assemble("llr", subject, impostor))
                print(
                    f"check, fold {fold.fold_index}, S{subject:03d} <- S{impostor:03d}: full "
                    f"replay of {full_starts.size} windows vs assembled: score max |diff| "
                    f"{score_diff.max():.2e}, llr max |diff| {llr_diff.max():.2e}"
                )

    print(
        f"windows per swap session: {int(before.sum())} genuine, {int(mixed.sum())} straddling "
        f"the splice, {int((~before & ~mixed).sum())} impostor"
    )
    impostor_rows = np.flatnonzero(claimed != source)
    result: dict[str, Pairs] = {}
    for unit in ("score", "llr"):
        sequences = [assemble(unit, int(claimed[i]), int(source[i])) for i in impostor_rows]
        result[unit] = (claimed[impostor_rows], source[impostor_rows], times, np.stack(sequences))

    quality = pair_quality(scores.protected)
    quality_rows = []
    for i in impostor_rows:
        subject, impostor = int(claimed[i]), int(source[i])
        session_ok = np.where(before, quality[row_of[(subject, subject)]], quality[i])
        session_ok[mixed] = straddling_ok[(subject, impostor)]
        quality_rows.append(session_ok)
    return result, np.stack(quality_rows)


def print_impostor_distribution(unit: str, group: str, values: Array, times: Array) -> None:
    digits = DIGITS[unit]
    width = digits + 6
    print(
        f"\nIMPOSTOR SESSIONS, {unit}, {group} ({values.shape[0]} sessions): percentiles of "
        "settled confidence, mean within-session SD, session maximum p10/p50/p90"
    )
    names = [f"p{p}" for p in PERCENTILES] + ["sessSD", "maxP10", "maxP50", "maxP90"]
    print(f"  {'':<8}" + "".join(f"{name:>{width}}" for name in names))
    for half_life in HALF_LIVES_S:
        confidence = ema_matrix(values, times, half_life)[:, settled_mask(times, half_life)]
        cells = [*np.percentile(confidence, PERCENTILES), confidence.std(axis=1).mean()]
        cells += list(np.percentile(confidence.max(axis=1), [10, 50, 90]))
        print(f"  {h_label(half_life):<8}" + "".join(f"{c:>{width}.{digits}f}" for c in cells))


def print_trajectories(unit: str, group: str, swaps: Array, times: Array) -> None:
    digits = DIGITS[unit]
    print(
        f"\nSWAP TRAJECTORIES, {unit}, {group} ({swaps.shape[0]} swaps at {SWAP_AT_S:g} s): "
        "confidence p50 [p10, p90] by seconds since the swap"
    )
    print(f"  {'t':>4}" + "".join(f"{h_label(h):>24}" for h in HALF_LIVES_S))
    confidence = {h: ema_matrix(swaps, times, h) for h in HALF_LIVES_S}
    for offset in TRAJECTORY_OFFSETS_S:
        column = int(np.flatnonzero(np.isclose(times, SWAP_AT_S + offset))[0])
        cells = []
        for h in HALF_LIVES_S:
            p10, p50, p90 = np.percentile(confidence[h][:, column], [10, 50, 90])
            cells.append(f"{p50:.{digits}f} [{p10:.{digits}f}, {p90:.{digits}f}]")
        print(f"  {offset:>+4}" + "".join(f"{cell:>24}" for cell in cells))


def print_crossings(
    unit: str, group: str, genuine: Array, impostor: Array, swaps: Array, times: Array
) -> None:
    print(
        f"\nCROSSING LEVELS, {unit}, {group}. genFalse% = genuine sessions whose settled "
        "confidence ever < level; armed% = swaps above level at the last pre-swap decision; "
        f"crossed% = armed swaps below level within {DETECTION_HORIZON_S:g} s; t50/t90 = seconds "
        "since the swap (inf = censored); neverCaught% = impostor sessions never < level"
    )
    header = ["h", "level", "genFalse%", "armed%", "crossed%", "t50", "t90", "neverCaught%"]
    widths = [8, 7, 11, 9, 10, 7, 7, 14]
    print(
        "  "
        + "".join(
            f"{name:>{w}}" if i else f"{name:<{w}}"
            for i, (name, w) in enumerate(zip(header, widths, strict=True))
        )
    )
    after = times > SWAP_AT_S
    last_before = int(np.flatnonzero(times <= SWAP_AT_S)[-1])
    for half_life in HALF_LIVES_S:
        settled = settled_mask(times, half_life)
        genuine_conf = ema_matrix(genuine, times, half_life)[:, settled]
        impostor_conf = ema_matrix(impostor, times, half_life)[:, settled]
        swap_conf = ema_matrix(swaps, times, half_life)
        for level in LEVELS[unit]:
            armed = swap_conf[:, last_before] >= level
            below = (swap_conf < level) & after
            delay = np.where(below.any(axis=1), times[below.argmax(axis=1)] - SWAP_AT_S, np.inf)
            delay = delay[armed]
            delay[delay > DETECTION_HORIZON_S] = np.inf
            if delay.size:
                crossed = 100.0 * float(np.isfinite(delay).mean())
                t50, t90 = np.quantile(delay, [0.5, 0.9], method="inverted_cdf")
            else:
                crossed, t50, t90 = float("nan"), float("nan"), float("nan")
            cells = [
                100.0 * float((genuine_conf < level).any(axis=1).mean()),
                100.0 * float(armed.mean()),
                crossed,
                t50,
                t90,
                100.0 * float((impostor_conf >= level).all(axis=1).mean()),
            ]
            print(
                f"  {h_label(half_life):<8}{level:>7.2f}{cells[0]:>11.1f}{cells[1]:>9.1f}"
                f"{cells[2]:>10.1f}{cells[3]:>7.0f}{cells[4]:>7.0f}{cells[5]:>14.1f}"
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cohort impostor and swap dynamics.")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    args = parser.parse_args(argv)
    try:
        scores = compute_cohort_scores(args.data_dir)
    except PreconditionError as exc:
        print(f"refusing to run: {exc}", file=sys.stderr)
        return 2
    for line in provenance_lines(scores):
        print(line)

    worst = np.array(sorted(scores.worst_decile), dtype=np.int64)
    swaps, _ = swap_sequences(scores, args.data_dir)
    delay = decision_delay(scores.streaming)
    for unit, table in (("score", scores.protected), ("llr", scores.decision)):
        claimed, source, onsets, values = pair_matrix(table)
        times = onsets + delay
        is_genuine = claimed == source
        swap_claimed, _, swap_times, swap_values = swaps[unit]
        for group, in_worst in (("worst decile", True), ("others", False)):
            in_group = np.isin(claimed, worst) == in_worst
            swap_in_group = np.isin(swap_claimed, worst) == in_worst
            impostor = values[~is_genuine & in_group]
            print_impostor_distribution(unit, group, impostor, times)
            print_trajectories(unit, group, swap_values[swap_in_group], swap_times)
            print_crossings(
                unit,
                group,
                values[is_genuine & in_group],
                impostor,
                swap_values[swap_in_group],
                swap_times,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
