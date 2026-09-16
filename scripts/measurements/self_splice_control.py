"""The pre-registered self-splice control, cohort only.

Impostor injection is simulated by splicing two recordings, so a swap carries a discontinuity
that a real electrode swap would not produce in the same way. Before any swap timing may be
reported as time-to-detect, that artifact has to be ruled out as the thing being detected
(D-022). A self-splice keeps the discontinuity and removes the identity change: the claimed
subject's own eyes-closed recording, spliced at SWAP_AT_S onto itself read from
SELF_SPLICE_OFFSET_S, then truncated to the original length so the session is the usual 56
decisions.

Criterion, fixed before this run (metrics.SPLICE_CONTROL_MAX_EXCESS = 0.10): self-splices may
not cross the revoke level within the detection horizon more often than genuine-only replays
do over the same stretch of session time, by more than 0.10 in absolute terms. Rates are
measured at the pre-registered parameters (D-025): protected score, half-life 4 s, revoke
0.56, challenge 0.58.

How a self-splice session is assembled, exactly as for the swaps: windows whose 6 s raw
context ends before the splice are the subject's own scored windows; windows whose context
begins after it are the same subject's scored windows from 20 s earlier, because splicing a
recording onto itself at a 10 s offset shifts the signal by exactly that much; the windows
straddling the splice are featurized here from the spliced signal. One subject per fold is
replayed in full and compared.

The impostor-swap rate over the same window is printed for contrast, not as part of the
criterion.

Everything is cohort only; see cohort_scores.py.

Usage:
    python -m scripts.measurements.self_splice_control
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from neuroauth.dsp.io import load_baseline_recordings
from neuroauth.dsp.streaming import context_window_starts, featurize_windows
from neuroauth.session.logic import PRE_REGISTERED_THRESHOLDS
from neuroauth.session.replay import CROSSFADE_S, DETECTION_HORIZON_S, SWAP_AT_S, splice
from neuroauth.templates.cancelable import derive_key, projection_matrix
from neuroauth.verification.metrics import SPLICE_CONTROL_MAX_EXCESS
from scripts.evaluate_verification import EVALUATION_MASTER_SECRET
from scripts.measurements.cohort_scores import (
    REPO_ROOT,
    CohortScores,
    compute_cohort_scores,
    provenance_lines,
)
from scripts.measurements.swap_dynamics import (
    Array,
    decision_delay,
    ema_matrix,
    pair_matrix,
    score_features,
    splice_parts,
    swap_sequences,
)
from scripts.train_baseline import PreconditionError

SELF_SPLICE_OFFSET_S = 10.0
"""Where the recording is read from after the splice. Not pre-registered: any offset other
than SWAP_AT_S keeps the discontinuity and the identity, and this one leaves the whole
post-splice stretch inside the recording."""


def self_splice_sequences(
    scores: CohortScores, data_dir: Path
) -> tuple[NDArray[np.int64], Array, Array]:
    """(claimed subjects, decision times, score sequences), one self-splice per enrolled subject."""
    streaming = scores.streaming
    claimed, source, onsets, values = pair_matrix(scores.protected)
    row_of = {(int(c), int(s)): i for i, (c, s) in enumerate(zip(claimed, source, strict=True))}
    times = onsets + decision_delay(streaming)
    shift_s = SWAP_AT_S - SELF_SPLICE_OFFSET_S
    onset_row = {round(float(onset), 6): i for i, onset in enumerate(onsets)}

    subjects: list[int] = []
    sequences: list[Array] = []
    for fold in scores.folds:
        enrolled = sorted(fold.templates)
        recordings = {
            r.subject_id: r
            for r in load_baseline_recordings(enrolled, data_dir, runs=(2,), skip_failures=False)
        }
        if set(recordings) & scores.holdout:
            raise AssertionError("a holdout subject's recording was loaded")
        sample = recordings[enrolled[0]]
        before, straddling, starts = splice_parts(onsets, sample.sfreq, streaming)
        after = ~(before | straddling)
        layer = scores.decision_layers[fold.fold_index]
        checked = False
        for subject in enrolled:
            template = fold.templates[subject]
            key = derive_key(EVALUATION_MASTER_SECRET, template.subject_ref, template.key_version)
            projection = projection_matrix(key, fold.model.n_components, fold.model.n_components)
            recording = recordings[subject]
            spliced = splice(
                recording,
                recording,
                swap_at_s=SWAP_AT_S,
                crossfade_s=CROSSFADE_S,
                second_offset_s=SELF_SPLICE_OFFSET_S,
            )[:, : recording.data.shape[1]]
            features = featurize_windows(
                spliced, starts[straddling], sample.sfreq, recording.ch_names, streaming
            )
            straddling_scores, _ = score_features(fold, layer, subject, features, projection)

            genuine = values[row_of[(subject, subject)]]
            sequence = genuine.copy()
            sequence[straddling] = straddling_scores
            for position in np.flatnonzero(after):
                sequence[position] = genuine[onset_row[round(float(onsets[position] - shift_s), 6)]]
            subjects.append(subject)
            sequences.append(sequence)

            if checked:
                continue
            checked = True
            full_starts = context_window_starts(spliced.shape[1], sample.sfreq, streaming)
            if not np.array_equal(full_starts, starts):
                raise ValueError("a full replay's windows differ from the table's windows")
            full = featurize_windows(
                spliced, full_starts, sample.sfreq, recording.ch_names, streaming
            )
            full_scores, _ = score_features(fold, layer, subject, full, projection)
            print(
                f"check, fold {fold.fold_index}, S{subject:03d} self-splice: full replay of "
                f"{full_starts.size} windows vs assembled: max |diff| "
                f"{np.abs(full_scores - sequence).max():.2e}"
            )
    return np.array(subjects, dtype=np.int64), times, np.stack(sequences)


def crossing_rate(sequences: Array, times: Array, level: float, half_life: float) -> float:
    """Share of sessions whose confidence drops below the level within the horizon window."""
    confidence = ema_matrix(sequences, times, half_life)
    inside = (times > SWAP_AT_S) & (times <= SWAP_AT_S + DETECTION_HORIZON_S)
    return float(np.mean((confidence[:, inside] < level).any(axis=1)))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Self-splice control, cohort only.")
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
    worst = np.array(sorted(scores.worst_decile), dtype=np.int64)
    self_subjects, times, self_sequences = self_splice_sequences(scores, args.data_dir)
    claimed, source, _, values = pair_matrix(scores.protected)
    genuine_rows = claimed == source
    genuine_subjects = claimed[genuine_rows]
    genuine_sequences = values[genuine_rows]
    swap_claimed, _, _, swap_values = swap_sequences(scores, args.data_dir)["score"]

    print(
        f"\nSELF-SPLICE CONTROL at the pre-registered parameters: score, h = {half_life:g} s, "
        f"revoke {thresholds.revoke_below:g}, challenge {thresholds.challenge_below:g}. "
        f"Rates are sessions crossing the level in ({SWAP_AT_S:g}, "
        f"{SWAP_AT_S + DETECTION_HORIZON_S:g}] s. Criterion: self-splice minus genuine "
        f"<= {SPLICE_CONTROL_MAX_EXCESS:g} at the revoke level."
    )
    header = ["level", "group", "n", "genuine%", "selfSplice%", "excess", "verdict", "swap%"]
    widths = [10, 14, 6, 10, 13, 9, 9, 8]
    print("  " + "".join(f"{name:>{w}}" for name, w in zip(header, widths, strict=True)))
    for name, level in (
        ("revoke", thresholds.revoke_below),
        ("challenge", thresholds.challenge_below),
    ):
        for group, keep in (
            ("all", None),
            ("worst decile", True),
            ("others", False),
        ):
            if keep is None:
                self_mask = np.ones(self_subjects.size, dtype=np.bool_)
                genuine_mask = np.ones(genuine_subjects.size, dtype=np.bool_)
                swap_mask = np.ones(swap_claimed.size, dtype=np.bool_)
            else:
                self_mask = np.isin(self_subjects, worst) == keep
                genuine_mask = np.isin(genuine_subjects, worst) == keep
                swap_mask = np.isin(swap_claimed, worst) == keep
            genuine_rate = crossing_rate(genuine_sequences[genuine_mask], times, level, half_life)
            self_rate = crossing_rate(self_sequences[self_mask], times, level, half_life)
            swap_rate = crossing_rate(swap_values[swap_mask], times, level, half_life)
            excess = self_rate - genuine_rate
            verdict = "PASS" if excess <= SPLICE_CONTROL_MAX_EXCESS else "FAIL"
            cells = [
                f"{name}",
                group,
                f"{int(self_mask.sum())}",
                f"{100 * genuine_rate:.1f}",
                f"{100 * self_rate:.1f}",
                f"{100 * excess:+.1f}",
                verdict,
                f"{100 * swap_rate:.1f}",
            ]
            print("  " + "".join(f"{c:>{w}}" for c, w in zip(cells, widths, strict=True)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
