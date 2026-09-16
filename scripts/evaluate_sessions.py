"""Run the Phase 2 session evaluation and write the committed artifacts.

Everything below was fixed before this script was written (D-016 rule; D-022, D-025):

    parameters  PRE_REGISTERED_THRESHOLDS: protected score, half-life 4 s, revoke 0.56,
                challenge 0.58, recover 0.62, dwell k = 3 within 8 s, flagged decisions skipped
    protocol    swap at SWAP_AT_S, 0.5 s raw crossfade, horizon DETECTION_HORIZON_S
    criteria    SESSION_DETECT_MAX_MEDIAN_S, SESSION_DETECT_MIN_CAUGHT,
                SESSION_MAX_GENUINE_REVOKE, SPLICE_CONTROL_MAX_EXCESS
    grouping    non-tail = outside the worst decile of cohort-impostor per-subject EER

Gates, all checked before anything is written:
    - config/impostor_holdout.json is committed and unmodified; the list is loaded from it
      (D-008). Writing artifacts/ requires a clean working tree.
    - every fold asserts training/fold/holdout disjointness inside score_fold (hard rules 1, 5).
    - every swap's injected impostor is asserted to be a holdout subject.

What is a holdout result here, and what is not (D-022). The swap criteria are: the impostor
spliced in is a holdout subject, unseen by the embedding, the decision layer, and the
thresholds. The genuine false-revoke criterion is not. Holdout subjects are never enrolled
(hard rule 1), so they cannot produce a genuine session, and genuine replays come from the
same enrollable subjects whose cohort replays the parameters were chosen on. That number
re-measures them through the real state machine instead of the EMA simulation in
scripts/measurements/dwell_cost.py, so a gap between the two is an implementation
discrepancy, not holdout degradation.

Sessions are assembled, not replayed frame by frame, which is what makes the run feasible at
89 x 20 swaps. The assembly is the one swap_dynamics.py verified against full replays of the
same spliced signal (max |diff| 0.00e+00 in every fold), and one session per fold is replayed
in full here too. The state machine itself is never simulated: every decision goes through
the real update_session.

"Over settled decisions" in the registered genuine criterion: the measurement scripts used
settled_mask, confidence at least one half-life (4 s) after the first decision. The state
machine reaches the same point by its own rule, min_scored_windows = 4 decisions, which at
the 1 s hop is the same 4 s. No transition can be reported before it.

Artifacts hold rates, counts, times, and versions only (hard rule 3).

Usage:
    python -m scripts.evaluate_sessions
    python -m scripts.evaluate_sessions --subjects 10 --impostors 4 --out <dir>  # partial
"""

import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mne
import numpy as np
import scipy
import sklearn
from numpy.typing import NDArray

from neuroauth.cohorts import DEFAULT_HOLDOUT_PATH, assert_holdout_excluded, load_holdout_record
from neuroauth.config import StreamingConfig
from neuroauth.dsp.io import load_baseline_recordings
from neuroauth.dsp.streaming import (
    check_margins,
    context_window_starts,
    featurize_windows,
    process_recording_bounded,
)
from neuroauth.dsp.types import FeatureMatrix
from neuroauth.session.logic import (
    PRE_REGISTERED_THRESHOLDS,
    TERMINAL_STATES,
    SessionThresholds,
    WindowObservation,
    initial_session_state,
    update_session,
)
from neuroauth.session.replay import CROSSFADE_S, DETECTION_HORIZON_S, SWAP_AT_S, splice
from neuroauth.templates.cancelable import derive_key, projection_matrix
from neuroauth.verification.decision import DecisionLayer
from neuroauth.verification.embedding import EmbeddingConfig
from neuroauth.verification.metrics import (
    SESSION_DETECT_MAX_MEDIAN_S,
    SESSION_DETECT_MIN_CAUGHT,
    SESSION_MAX_GENUINE_REVOKE,
    SPLICE_CONTROL_MAX_EXCESS,
    DetectionSummary,
    FalseTransitionSummary,
    ScoreTable,
    SessionStateName,
    SessionTrace,
    concatenate_tables,
    eer_value,
    equal_error_rate,
    error_rates,
    false_transition_rate,
    genuine_mask,
    impostor_mask,
    per_subject_eer,
    summarize_tail,
    summarize_time_to_detect,
)
from neuroauth.verification.protocol import (
    FOLD_SEED,
    N_FOLDS,
    FoldScores,
    cross_fit_decision_layers,
    cross_fit_decision_table,
    score_fold,
    subject_folds,
)
from scripts.evaluate_verification import EVALUATION_MASTER_SECRET
from scripts.measurements.swap_dynamics import (
    Array,
    Mask,
    decision_delay,
    pair_matrix,
    pair_quality,
    score_features,
    splice_parts,
)
from scripts.train_baseline import PreconditionError, check_holdout_committed, working_tree_changes

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "artifacts" / "sessions"
SPLIT = "cross_condition"
SELF_SPLICE_OFFSET_S = 10.0
"""Matches scripts/measurements/self_splice_control.py, so the control measures the same
thing here as it did on cohort data."""


@dataclass(frozen=True)
class SessionScores:
    """Fold models and score tables for the session run, holdout rows included.

    Attributes:
        streaming: Settings the features were built with.
        holdout: The committed impostor holdout.
        folds: Per fold, its model, templates, and tables.
        decision_layers: Fold -> the cross-fitted decision layer that scores that fold.
        protected: Hamming similarity, every claimed/source pair.
        decision: Decision layer v0 LLR, cross-fitted.
        per_subject_eer: Claimed subject -> EER against cohort impostors only.
        worst_decile: Worst decile by that EER. Ranked on cohort scores, never on the
            session results being judged (D-022).
        failed_to_enroll: Subjects with too few enrollment windows.
    """

    streaming: StreamingConfig
    holdout: frozenset[int]
    folds: tuple[FoldScores, ...]
    decision_layers: dict[int, DecisionLayer]
    protected: ScoreTable
    decision: ScoreTable
    per_subject_eer: dict[int, float]
    worst_decile: tuple[int, ...]
    failed_to_enroll: tuple[int, ...]


@dataclass(frozen=True)
class SessionSet:
    """Assembled sessions of one kind, one row per session.

    Attributes:
        kind: "genuine", "swap", or "self_splice".
        claimed: Enrolled identity each session was opened for.
        impostor: Source after the swap, or None for genuine-only replays.
        times: Decision times, shared by every session.
        scores: Protected similarity per (session, decision).
        llrs: Decision layer v0 LLR, carried for logging only (D-025).
        quality: probe_window_ok per (session, decision).
        swap_at_s: Swap time, or None for genuine-only.
    """

    kind: str
    claimed: NDArray[np.int64]
    impostor: NDArray[np.int64] | None
    times: Array
    scores: Array
    llrs: Array
    quality: Mask
    swap_at_s: float | None


def compute_session_scores(
    matrices: list[FeatureMatrix],
    *,
    enrollable: tuple[int, ...],
    holdout: frozenset[int],
    streaming: StreamingConfig,
) -> SessionScores:
    """Score every fold on the headline split, with holdout subjects as impostor probes."""
    folds = subject_folds(enrollable, N_FOLDS, FOLD_SEED)
    enrolled_at = datetime.now(UTC)
    fold_scores = tuple(
        score_fold(
            matrices,
            fold_index=index,
            fold_subjects=frozenset(fold),
            train_subjects=frozenset(enrollable) - frozenset(fold),
            impostor_holdout=holdout,
            split_kind=SPLIT,
            streaming=streaming,
            embedding_config=EmbeddingConfig(),
            master_secret=EVALUATION_MASTER_SECRET,
            enrolled_at=enrolled_at,
        )
        for index, fold in enumerate(folds)
    )
    protected = concatenate_tables([f.protected_table for f in fold_scores])
    n_bits = min(f.model.n_components for f in fold_scores)
    versions = {f.fold_index: f.representation_version for f in fold_scores}
    decision, _ = cross_fit_decision_table(
        protected, n_bits=n_bits, representation_versions=versions
    )
    layers = cross_fit_decision_layers(protected, n_bits=n_bits, representation_versions=versions)

    per_subject = per_subject_eer(protected, "cohort")
    pooled = eer_value(
        equal_error_rate(
            error_rates(
                protected.scores[genuine_mask(protected)],
                protected.scores[impostor_mask(protected, "cohort")],
            )
        )
    )
    tail = summarize_tail(per_subject, pooled_eer=pooled)
    return SessionScores(
        streaming=streaming,
        holdout=holdout,
        folds=fold_scores,
        decision_layers=layers,
        protected=protected,
        decision=decision,
        per_subject_eer=per_subject,
        worst_decile=tail.worst_decile_subjects,
        failed_to_enroll=tuple(sorted(s for f in fold_scores for s in f.failed_to_enroll)),
    )


def session_states(
    scores: Array,
    llrs: Array,
    quality: Mask,
    times: Array,
    thresholds: SessionThresholds,
) -> tuple[SessionStateName, ...]:
    """Fold the real update_session over one session's decisions.

    Stops at the first terminal state, as the runtime does: after revocation it stops scoring
    and closes the socket, so no further decision exists.
    """
    state = initial_session_state()
    states: list[SessionStateName] = []
    for score, llr, ok, decision_time in zip(scores, llrs, quality, times, strict=True):
        observation = WindowObservation(
            decision_time_s=float(decision_time),
            score=float(score),
            llr=float(llr),
            quality_ok=bool(ok),
            gap_before=False,
        )
        state, _ = update_session(state, observation, thresholds)
        states.append(state.state)
        if state.state in TERMINAL_STATES:
            break
    return tuple(states)


def build_traces(sessions: SessionSet, thresholds: SessionThresholds) -> list[SessionTrace]:
    """One SessionTrace per assembled session, driven through the real state machine."""
    traces: list[SessionTrace] = []
    for row in range(sessions.claimed.size):
        states = session_states(
            sessions.scores[row],
            sessions.llrs[row],
            sessions.quality[row],
            sessions.times,
            thresholds,
        )
        impostor = None if sessions.impostor is None else int(sessions.impostor[row])
        traces.append(
            SessionTrace(
                claimed_subject=int(sessions.claimed[row]),
                decision_times_s=sessions.times[: len(states)],
                states=states,
                swap_at_s=sessions.swap_at_s,
                impostor_subject=impostor,
            )
        )
    return traces


def non_tail_mask(claimed: NDArray[np.int64], worst_decile: NDArray[np.int64]) -> Mask:
    """Sessions whose claimed subject is outside the worst decile of cohort per-subject EER."""
    mask: Mask = ~np.isin(claimed, worst_decile)
    return mask


def select(sessions: SessionSet, mask: Mask) -> SessionSet:
    return SessionSet(
        kind=sessions.kind,
        claimed=sessions.claimed[mask],
        impostor=None if sessions.impostor is None else sessions.impostor[mask],
        times=sessions.times,
        scores=sessions.scores[mask],
        llrs=sessions.llrs[mask],
        quality=sessions.quality[mask],
        swap_at_s=sessions.swap_at_s,
    )


def genuine_sessions(scores: SessionScores) -> SessionSet:
    """Each enrolled subject's own eyes-closed recording, replayed against their template."""
    claimed, source, onsets, values = pair_matrix(scores.protected)
    _, _, _, llrs = pair_matrix(scores.decision)
    quality = pair_quality(scores.protected)
    rows = claimed == source
    return SessionSet(
        kind="genuine",
        claimed=claimed[rows],
        impostor=None,
        times=onsets + decision_delay(scores.streaming),
        scores=values[rows],
        llrs=llrs[rows],
        quality=quality[rows],
        swap_at_s=None,
    )


def _fold_projection(fold: FoldScores, subject: int) -> Array:
    template = fold.templates[subject]
    key = derive_key(EVALUATION_MASTER_SECRET, template.subject_ref, template.key_version)
    projection: Array = projection_matrix(key, fold.model.n_components, fold.model.n_components)
    return projection


def swap_sessions(scores: SessionScores, data_dir: Path) -> SessionSet:
    """Each enrolled subject's recording spliced onto a holdout impostor's at SWAP_AT_S.

    Windows whose 6 s raw context ends before the splice are the claimed subject's genuine
    windows; windows whose context begins after it are the impostor's windows scored against
    the claimed template; the windows straddling the splice are featurized here from the
    spliced signal. One swap per fold is replayed in full and compared.

    Raises:
        AssertionError: If an injected impostor is not a holdout subject.
    """
    streaming = scores.streaming
    claimed, source, onsets, values = pair_matrix(scores.protected)
    _, _, _, llr_values = pair_matrix(scores.decision)
    quality = pair_quality(scores.protected)
    row_of = {(int(c), int(s)): i for i, (c, s) in enumerate(zip(claimed, source, strict=True))}
    times = onsets + decision_delay(streaming)
    impostors = sorted(scores.holdout)

    out_claimed: list[int] = []
    out_impostor: list[int] = []
    sequences: list[Array] = []
    llr_sequences: list[Array] = []
    qualities: list[Mask] = []
    for fold in scores.folds:
        enrolled = sorted(fold.templates)
        recordings = {
            r.subject_id: r
            for r in load_baseline_recordings(
                sorted({*enrolled, *impostors}), data_dir, runs=(2,), skip_failures=False
            )
        }
        sample = recordings[enrolled[0]]
        before, straddling, starts = splice_parts(onsets, sample.sfreq, streaming)
        layer = scores.decision_layers[fold.fold_index]
        checked = False
        for subject in enrolled:
            projection = _fold_projection(fold, subject)
            for impostor in impostors:
                if impostor not in scores.holdout:
                    raise AssertionError(f"S{impostor:03d} is not a holdout subject")
                spliced = splice(
                    recordings[subject],
                    recordings[impostor],
                    swap_at_s=SWAP_AT_S,
                    crossfade_s=CROSSFADE_S,
                )
                features = featurize_windows(
                    spliced, starts[straddling], sample.sfreq, sample.ch_names, streaming
                )
                mixed_score, mixed_llr = score_features(fold, layer, subject, features, projection)
                genuine_row = row_of[(subject, subject)]
                impostor_row = row_of[(subject, impostor)]
                sequence = np.where(before, values[genuine_row], values[impostor_row])
                llr_sequence = np.where(before, llr_values[genuine_row], llr_values[impostor_row])
                session_ok = np.where(before, quality[genuine_row], quality[impostor_row])
                sequence[straddling] = mixed_score
                llr_sequence[straddling] = mixed_llr
                session_ok[straddling] = features.quality.window_ok

                out_claimed.append(subject)
                out_impostor.append(impostor)
                sequences.append(sequence)
                llr_sequences.append(llr_sequence)
                qualities.append(session_ok)

                if checked:
                    continue
                checked = True
                full_starts = context_window_starts(spliced.shape[1], sample.sfreq, streaming)
                if not np.array_equal(full_starts, starts):
                    raise ValueError("a full replay's windows differ from the table's windows")
                full = featurize_windows(
                    spliced, full_starts, sample.sfreq, sample.ch_names, streaming
                )
                full_score, _ = score_features(fold, layer, subject, full, projection)
                print(
                    f"  check, fold {fold.fold_index}, S{subject:03d} <- S{impostor:03d}: full "
                    f"replay of {full_starts.size} windows vs assembled: max |diff| "
                    f"{np.abs(full_score - sequence).max():.2e}",
                    flush=True,
                )
    return SessionSet(
        kind="swap",
        claimed=np.array(out_claimed, dtype=np.int64),
        impostor=np.array(out_impostor, dtype=np.int64),
        times=times,
        scores=np.stack(sequences),
        llrs=np.stack(llr_sequences),
        quality=np.stack(qualities),
        swap_at_s=SWAP_AT_S,
    )


def self_splice_sessions(scores: SessionScores, data_dir: Path) -> SessionSet:
    """Each subject's own recording spliced onto itself at a different offset.

    Keeps the discontinuity, removes the identity change. Assembled exactly as in
    scripts/measurements/self_splice_control.py: post-splice windows are the same subject's
    windows from SWAP_AT_S - SELF_SPLICE_OFFSET_S earlier.
    """
    streaming = scores.streaming
    claimed, source, onsets, values = pair_matrix(scores.protected)
    _, _, _, llr_values = pair_matrix(scores.decision)
    quality = pair_quality(scores.protected)
    row_of = {(int(c), int(s)): i for i, (c, s) in enumerate(zip(claimed, source, strict=True))}
    times = onsets + decision_delay(streaming)
    shift_s = SWAP_AT_S - SELF_SPLICE_OFFSET_S
    onset_row = {round(float(onset), 6): i for i, onset in enumerate(onsets)}

    subjects: list[int] = []
    sequences: list[Array] = []
    llr_sequences: list[Array] = []
    qualities: list[Mask] = []
    for fold in scores.folds:
        enrolled = sorted(fold.templates)
        recordings = {
            r.subject_id: r
            for r in load_baseline_recordings(enrolled, data_dir, runs=(2,), skip_failures=False)
        }
        if set(recordings) & scores.holdout:
            raise AssertionError("a holdout subject cannot produce a genuine session")
        sample = recordings[enrolled[0]]
        before, straddling, starts = splice_parts(onsets, sample.sfreq, streaming)
        after = ~(before | straddling)
        layer = scores.decision_layers[fold.fold_index]
        for subject in enrolled:
            projection = _fold_projection(fold, subject)
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
            mixed_score, mixed_llr = score_features(fold, layer, subject, features, projection)

            row = row_of[(subject, subject)]
            sequence = values[row].copy()
            llr_sequence = llr_values[row].copy()
            session_ok = quality[row].copy()
            sequence[straddling] = mixed_score
            llr_sequence[straddling] = mixed_llr
            session_ok[straddling] = features.quality.window_ok
            for position in np.flatnonzero(after):
                origin = onset_row[round(float(onsets[position] - shift_s), 6)]
                sequence[position] = values[row][origin]
                llr_sequence[position] = llr_values[row][origin]
                session_ok[position] = quality[row][origin]

            subjects.append(subject)
            sequences.append(sequence)
            llr_sequences.append(llr_sequence)
            qualities.append(session_ok)
    claimed_out = np.array(subjects, dtype=np.int64)
    return SessionSet(
        kind="self_splice",
        claimed=claimed_out,
        impostor=claimed_out,
        times=times,
        scores=np.stack(sequences),
        llrs=np.stack(llr_sequences),
        quality=np.stack(qualities),
        swap_at_s=SWAP_AT_S,
    )


def judge(
    detection: DetectionSummary,
    genuine: FalseTransitionSummary,
    self_splice: FalseTransitionSummary,
) -> dict[str, Any]:
    """Apply the registered criteria (D-016, D-022). Failure is reported, never retuned."""
    excess = self_splice.fraction_of_sessions - genuine.fraction_of_sessions
    splice_passes = excess <= SPLICE_CONTROL_MAX_EXCESS
    return {
        "median_time_to_detect_s": {
            "value": detection.median_s,
            "criterion": f"<= {SESSION_DETECT_MAX_MEDIAN_S:g}",
            "passes": detection.median_s <= SESSION_DETECT_MAX_MEDIAN_S,
        },
        "fraction_caught_within_horizon": {
            "value": detection.fraction_detected_within_horizon,
            "criterion": f">= {SESSION_DETECT_MIN_CAUGHT:g}",
            "passes": detection.fraction_detected_within_horizon >= SESSION_DETECT_MIN_CAUGHT,
        },
        "genuine_sessions_revoked": {
            "value": genuine.fraction_of_sessions,
            "criterion": f"<= {SESSION_MAX_GENUINE_REVOKE:g}",
            "passes": genuine.fraction_of_sessions <= SESSION_MAX_GENUINE_REVOKE,
            "is_holdout_result": False,
            "note": (
                "Holdout subjects are never enrolled, so genuine replays are the same "
                "enrollable subjects the parameters were chosen on. A gap against the "
                "cohort measurement is an implementation discrepancy, not degradation."
            ),
        },
        "self_splice_excess": {
            "value": excess,
            "criterion": f"<= {SPLICE_CONTROL_MAX_EXCESS:g}",
            "passes": splice_passes,
        },
        "time_to_detect_reportable": splice_passes,
    }


def _summary_block(
    sessions: SessionSet, traces: list[SessionTrace], *, swap: bool
) -> dict[str, Any]:
    block: dict[str, Any] = {"n_sessions": len(traces), "kind": sessions.kind}
    if swap:
        detection = summarize_time_to_detect(traces, "revoked", horizon_s=DETECTION_HORIZON_S)
        challenge = summarize_time_to_detect(traces, "challenged", horizon_s=DETECTION_HORIZON_S)
        block["revoked"] = {
            "median_s": detection.median_s,
            "p90_s": detection.p90_s,
            "fraction_within_horizon": detection.fraction_detected_within_horizon,
            "n_censored": detection.n_censored,
            "horizon_s": detection.horizon_s,
        }
        block["challenged"] = {
            "median_s": challenge.median_s,
            "fraction_within_horizon": challenge.fraction_detected_within_horizon,
        }
    else:
        for target in ("revoked", "challenged"):
            rate = false_transition_rate(traces, target)  # type: ignore[arg-type]
            block[target] = {
                "fraction_of_sessions": rate.fraction_of_sessions,
                "events_per_hour": rate.events_per_hour,
                "exposure_s": rate.exposure_s,
            }
    return block


def run(
    scores: SessionScores,
    data_dir: Path,
    *,
    out_dir: Path,
    thresholds: SessionThresholds,
    provenance: dict[str, Any],
    partial: bool,
) -> dict[str, Any]:
    """Assemble every session kind, drive the state machine, judge, and only then write."""
    worst = np.array(sorted(scores.worst_decile), dtype=np.int64)
    print("assembling genuine sessions", flush=True)
    genuine = genuine_sessions(scores)
    print("assembling swaps against holdout impostors", flush=True)
    swaps = swap_sessions(scores, data_dir)
    print("assembling the self-splice control", flush=True)
    self_splice = self_splice_sessions(scores, data_dir)

    groups: dict[str, Any] = {}
    for group, keep in (("non_tail", True), ("worst_decile", False), ("all", None)):
        selected = {}
        for sessions in (genuine, swaps, self_splice):
            mask = (
                np.ones(sessions.claimed.size, dtype=np.bool_)
                if keep is None
                else non_tail_mask(sessions.claimed, worst) == keep
            )
            selected[sessions.kind] = select(sessions, mask)
        traces = {kind: build_traces(s, thresholds) for kind, s in selected.items()}
        block = {
            kind: _summary_block(selected[kind], traces[kind], swap=kind != "genuine")
            for kind in selected
        }
        if group == "non_tail":
            block["verdicts"] = judge(
                summarize_time_to_detect(traces["swap"], "revoked", horizon_s=DETECTION_HORIZON_S),
                false_transition_rate(traces["genuine"], "revoked"),
                false_transition_rate(traces["self_splice"], "revoked"),
            )
        groups[group] = block

    summary = {
        **provenance,
        "what_this_is": (
            "Phase 2 session evaluation: swap detection against held-out impostors, genuine "
            "false-revoke, and the self-splice control, all through the real update_session. "
            "Swap criteria are holdout results; genuine false-revoke is a re-measurement on "
            "the subjects the parameters were chosen from (D-022)."
        ),
        "partial_run": partial,
        "groups": groups,
        "worst_decile_subjects": list(scores.worst_decile),
        "worst_decile_ranked_on": "cohort-impostor per-subject EER, never the session results",
        "failed_to_enroll": list(scores.failed_to_enroll),
        "streaming_fingerprint": scores.streaming.fingerprint(),
        "a_priori": {
            "session_thresholds": {
                "ema_half_life_s": thresholds.ema_half_life_s,
                "challenge_below": thresholds.challenge_below,
                "revoke_below": thresholds.revoke_below,
                "recover_above": thresholds.recover_above,
                "revoke_dwell_decisions": thresholds.revoke_dwell_decisions,
                "revoke_dwell_max_span_s": thresholds.revoke_dwell_max_span_s,
                "min_scored_windows": thresholds.min_scored_windows,
                "max_consecutive_not_ok": thresholds.max_consecutive_not_ok,
            },
            "session_detect_max_median_s": SESSION_DETECT_MAX_MEDIAN_S,
            "session_detect_min_caught": SESSION_DETECT_MIN_CAUGHT,
            "session_max_genuine_revoke": SESSION_MAX_GENUINE_REVOKE,
            "splice_control_max_excess": SPLICE_CONTROL_MAX_EXCESS,
            "swap_at_s": SWAP_AT_S,
            "crossfade_s": CROSSFADE_S,
            "detection_horizon_s": DETECTION_HORIZON_S,
            "n_folds": N_FOLDS,
            "fold_seed": FOLD_SEED,
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "session_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 2 session evaluation.")
    parser.add_argument("--subjects", type=int, default=None, help="first N enrollable subjects")
    parser.add_argument("--impostors", type=int, default=None, help="first M holdout subjects")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--holdout", type=Path, default=REPO_ROOT / DEFAULT_HOLDOUT_PATH)
    args = parser.parse_args(argv)
    if (args.subjects is not None or args.impostors is not None) and args.out is None:
        parser.error(
            "--subjects/--impostors make a partial run; pass --out so artifacts/ is untouched"
        )
    return args


def main(argv: list[str] | None = None) -> int:
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
        scores = compute_session_scores(
            matrices,
            enrollable=tuple(enrollable),
            holdout=frozenset(impostors),
            streaming=streaming,
        )
        summary = run(
            scores,
            args.data_dir,
            out_dir=out_dir,
            thresholds=PRE_REGISTERED_THRESHOLDS,
            provenance=provenance,
            partial=args.subjects is not None or args.impostors is not None,
        )
    except AssertionError as exc:
        print(f"aborted, no artifacts written: {exc}", file=sys.stderr)
        return 1

    verdicts = summary["groups"]["non_tail"]["verdicts"]
    print("\nNON-TAIL SUBJECTS, registered criteria (D-016, D-022):")
    for name, block in verdicts.items():
        if not isinstance(block, dict):
            continue
        mark = "PASS" if block["passes"] else "FAIL"
        print(f"  {mark}  {name}: {block['value']:.3f}, criterion {block['criterion']}")
    if not verdicts["time_to_detect_reportable"]:
        print("  self-splice control failed: time-to-detect is not reported as a result (D-022)")
    print(f"\nartifacts written to {out_dir} in {time.perf_counter() - started:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
