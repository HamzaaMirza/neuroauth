"""Biometric error rates, the per-subject tail, and time-to-detect. No accuracy, anywhere.

Hard rule 2 and D-012: no report here has an accuracy field, and a test pins that.

Every constant below judges a result. They are proposed in the Phase 2 contracts and fixed
when this file is committed, before any Phase 2 result exists (D-016 rule). A test pins
them once implemented.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray

ScoreDomain = Literal["embedding", "protected"]
"""embedding: cosine similarity of unprotected embeddings, computed in memory only, to
measure the cost of protection. protected: Hamming similarity of BioHash bits, which is the
system's actual score and the headline."""

ImpostorSource = Literal["impostor_holdout", "cohort"]
"""impostor_holdout: the 20 committed subjects, used only for reporting. cohort: other
enrollable subjects in the same fold, never fitted on in that fold. Used for choosing
thresholds, and never reported as the headline FAR."""

SessionStateName = Literal["active", "challenged", "revoked", "expired", "closed"]

# --------------------------------------------------------------- a priori thresholds

FAR_TARGETS: Final = (0.01, 0.001)
"""Operating points reported as FRR at FAR. Both are always reported; each carries its own
under-resolution flag."""

HEADLINE_FAR: Final = 0.01
"""The headline operating point is FRR at FAR = 0.01.

FAR = 0.001 is the roadmap's number and is still reported, but at 1780 pairs it is
under-resolved (MIN_EXPECTED_ERRORS), and a headline must be a number the data can support.
If the headline point is itself under-resolved (fewer than 300 pairs, e.g. after enrollment
failures), the report says so. It does not move to a different operating point."""

MIN_EXPECTED_ERRORS: Final = 3.0
"""An operating point is under-resolved when n_impostor_pairs * target_far is below this
(rule of three). Counted in (claimed subject, impostor subject) pairs, not windows, because
one impostor's windows against one template are not independent. With 89 x 20 = 1780 pairs,
FAR 0.01 expects 17.8 errors and is resolved; FAR 0.001 expects 1.8 and is under-resolved.
No split of 109 subjects could resolve it, since (109 - n) * n <= 2970 < 3000."""

RANDOM_PAIRING_CONTROL_MIN_EER: Final = 0.40
"""With genuine scores replaced by scores against a different enrolled subject, pooled EER
must be at least this, or the run writes nothing. Do not re-run seeds to pass it (D-007
posture)."""

PROTECTION_MATERIAL_EER_COST: Final = 0.02
"""Protected-domain EER above embedding-domain EER by more than this is reported as a
material cost of template protection."""

TAIL_WORST_FRACTION: Final = 0.10
TAIL_HEAVY_MIN_RATIO: Final = 2.0
TAIL_HEAVY_MIN_GAP: Final = 0.10
"""Prediction P1a (PROGRESS, 2026-09-14): per-subject EER has a heavy tail. Confirmed iff
the worst-decile mean is at least 2x the median AND at least 0.10 above it. Otherwise
refuted."""

CONCORDANCE_CONFIRM_MAX_RHO: Final = -0.30
CONCORDANCE_REFUTE_MIN_RHO: Final = -0.10
CONCORDANCE_ALPHA: Final = 0.05
"""Prediction P1b: the subjects in Phase 1's F1 tail are the subjects in Phase 2's EER tail.
Spearman rho between Phase 1 per-subject F1 and Phase 2 per-subject EER. Confirmed iff
rho <= -0.30 and the one-sided permutation p < 0.05. Refuted iff rho > -0.10. Otherwise
inconclusive."""

REVOCATION_AGREEMENT_BAND: Final = (0.45, 0.55)
REVOCATION_MAX_ACCEPTED_FRACTION: Final = 0.03
"""Revocation works iff (a) the mean bit agreement between each subject's revoked and
reissued templates lies in the band, and (b) at most 3% of subjects' revoked templates,
presented as probes, are accepted by their reissued template at the cohort-selected
FAR = 0.01 threshold."""

SPLICE_CONTROL_MAX_EXCESS: Final = 0.10
"""Impostor injection is simulated by splicing recordings. Self-splices (a subject spliced
to a later part of their own probe recording) must not be revoked within the horizon more
often than genuine-only replays by more than this absolute fraction. Otherwise the splice
artifact, not identity, is driving detection, and time-to-detect is not reported as a
result."""

N_BOOTSTRAP: Final = 2000
BOOTSTRAP_SEED: Final = 20260916
N_PERMUTATIONS: Final = 10000
PERMUTATION_SEED: Final = 20260917
PAIRING_CONTROL_SEED: Final = 20260918


# --------------------------------------------------------------- score tables


@dataclass(frozen=True)
class ScoreTable:
    """One row per (probe window, claimed identity) comparison. Scores only, so persistable.

    A row is genuine iff claimed_subject == source_subject.

    Attributes:
        scores: (n,) accept iff score >= threshold.
        claimed_subject: (n,) enrolled subject whose template was used.
        source_subject: (n,) subject whose EEG the probe window is.
        source_is_holdout: (n,) source_subject is in the impostor holdout.
        fold: (n,) fold of the claimed subject, identifying the model used.
        probe_onset_s: (n,) onset in the probe recording.
        probe_window_ok: (n,) quality mask of the probe window.
        domain: Which score.
        split_kind: "cross_condition" (headline) or "temporal".
    """

    scores: NDArray[np.float64]
    claimed_subject: NDArray[np.int64]
    source_subject: NDArray[np.int64]
    source_is_holdout: NDArray[np.bool_]
    fold: NDArray[np.int64]
    probe_onset_s: NDArray[np.float64]
    probe_window_ok: NDArray[np.bool_]
    domain: ScoreDomain
    split_kind: str


def check_score_table(table: ScoreTable, impostor_holdout: frozenset[int]) -> None:
    """Structural and cohort integrity, run on every table before any metric.

    Raises:
        AssertionError: If any claimed_subject is in the holdout (never enrolled),
            source_is_holdout disagrees with the holdout set, a genuine row has a holdout
            source, or any score is non-finite.
        ValueError: If column lengths differ.
    """
    raise NotImplementedError("TODO(phase-2): score table integrity")


def split_scores(
    table: ScoreTable,
    impostor_source: ImpostorSource,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """(genuine, impostor) scores, with impostor rows restricted to one source."""
    raise NotImplementedError("TODO(phase-2): select genuine and impostor rows")


# --------------------------------------------------------------- error rates


@dataclass(frozen=True)
class ErrorRates:
    """FAR and FRR at every distinct threshold: the DET curve before transformation.

    Attributes:
        thresholds: (k,) ascending. Every distinct observed score, then +inf.
        far: (k,) fraction of impostor scores >= threshold. Non-increasing; 0 at +inf.
        frr: (k,) fraction of genuine scores < threshold. Non-decreasing; 1 at +inf.
        n_genuine: Genuine scores.
        n_impostor: Impostor scores.
    """

    thresholds: NDArray[np.float64]
    far: NDArray[np.float64]
    frr: NDArray[np.float64]
    n_genuine: int
    n_impostor: int


def error_rates(genuine: NDArray[np.float64], impostor: NDArray[np.float64]) -> ErrorRates:
    """FAR and FRR at every distinct threshold, with accept iff score >= threshold.

    Tied scores are accepted or rejected together. No operating point between ties is
    invented, which matters because Hamming similarity moves in steps of 1/64.

    Raises:
        ValueError: If either side is empty, not 1-D, or non-finite. This is evaluation,
            not inference, so bad input raises.
    """
    raise NotImplementedError("TODO(phase-2): FAR/FRR sweep")


@dataclass(frozen=True)
class OperatingPoint:
    """A threshold and the error rates it achieves on the scores it was evaluated on.

    Attributes:
        threshold: Accept iff score >= threshold.
        far: Achieved FAR.
        frr: Achieved FRR.
        target_far: Requested FAR, or None for the EER point.
        under_resolved: n_impostor_pairs * target_far < MIN_EXPECTED_ERRORS, or None for
            the EER point. The rates are still reported, and the flag goes with them into
            every artifact and README table.
    """

    threshold: float
    far: float
    frr: float
    target_far: float | None
    under_resolved: bool | None


def equal_error_rate(rates: ErrorRates) -> OperatingPoint:
    """EER as the smallest achievable max(FAR, FRR), with the threshold that achieves it.

    Chosen over interpolating the FAR/FRR crossing, which reports a rate that no single
    threshold attains, and over the ROC-convex-hull EER, which is harder to explain for no
    gain at these sample sizes. It can exceed the interpolated value by at most one step,
    max(1/n_genuine, 1/n_impostor). The returned far and frr are both reported, and EER is
    their maximum.
    """
    raise NotImplementedError("TODO(phase-2): minimax EER")


def frr_at_far(rates: ErrorRates, target_far: float, *, n_impostor_pairs: int) -> OperatingPoint:
    """Lowest FRR subject to FAR <= target_far.

    This is the smallest threshold whose FAR does not exceed the target. If no impostor
    score can be excluded without rejecting everything, the threshold is +inf with FRR 1,
    reported rather than raised.

    Args:
        rates: From error_rates.
        target_far: In (0, 1).
        n_impostor_pairs: Distinct (claimed, impostor subject) pairs behind the impostor
            scores, for the under_resolved flag.

    Raises:
        ValueError: If target_far is outside (0, 1) or n_impostor_pairs is below 1.
    """
    raise NotImplementedError("TODO(phase-2): FRR at fixed FAR")


def rates_at_threshold(
    genuine: NDArray[np.float64],
    impostor: NDArray[np.float64],
    threshold: float,
) -> OperatingPoint:
    """FAR and FRR at a threshold chosen elsewhere.

    This is how a threshold selected on cohort scores is reported on the holdout. It is the
    deployable number, as opposed to the oracle EER.
    """
    raise NotImplementedError("TODO(phase-2): rates at a fixed threshold")


def det_coordinates(rates: ErrorRates) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Probit-transformed (FAR, FRR) for a DET plot.

    Rates of exactly 0 or 1 are clipped to 0.5/n and 1 - 0.5/n of their side before the
    inverse normal CDF, and the plot marks the clipped region.
    """
    raise NotImplementedError("TODO(phase-2): probit DET coordinates")


def write_det_plot(
    curves: Mapping[str, ErrorRates],
    path: Path,
    *,
    title: str,
    eer_points: Mapping[str, OperatingPoint],
) -> None:
    """Write a DET plot, one curve per label, with EER and FAR-target markers.

    Axes are labelled in rates (0.1%, 1%, 10%) on probit scales. Styled like the Phase 1
    confusion matrix.
    """
    raise NotImplementedError("TODO(phase-2): DET plot artifact")


# --------------------------------------------------------------- per-subject tail


def per_subject_eer(table: ScoreTable, impostor_source: ImpostorSource) -> dict[int, float]:
    """EER per claimed subject, from its genuine scores and its impostor scores.

    Uses a per-subject threshold, which makes it optimistic compared with the pooled EER's
    single global threshold. Both are reported and never conflated.

    Returns:
        Claimed subject -> EER, for every claimed subject in the table.
    """
    raise NotImplementedError("TODO(phase-2): per-subject EER")


@dataclass(frozen=True)
class TailSummary:
    """Distribution of per-subject EER, with the P1a verdict.

    Attributes:
        n_subjects: Subjects summarized.
        median: Median per-subject EER.
        p90: 90th percentile.
        worst_decile_mean: Mean EER of the worst ceil(0.1 * n) subjects.
        worst_decile_subjects: Those subjects, worst first, ties broken by ascending id.
        pooled_eer: Global-threshold EER on the same table, for contrast.
        heavy_tail: The P1a verdict under TAIL_HEAVY_MIN_RATIO and TAIL_HEAVY_MIN_GAP.
    """

    n_subjects: int
    median: float
    p90: float
    worst_decile_mean: float
    worst_decile_subjects: tuple[int, ...]
    pooled_eer: float
    heavy_tail: bool


def summarize_tail(per_subject: Mapping[int, float], *, pooled_eer: float) -> TailSummary:
    """Summarize per-subject EER and apply the a priori heavy-tail rule.

    Raises:
        ValueError: If per_subject is empty.
    """
    raise NotImplementedError("TODO(phase-2): tail summary")


@dataclass(frozen=True)
class ConcordanceResult:
    """P1b: do Phase 1's worst-identified subjects become Phase 2's worst-verified?

    Attributes:
        spearman_rho: Between Phase 1 F1 and Phase 2 EER, over subjects in both.
        p_value: One-sided (rho < 0) permutation p-value.
        n_subjects: Subjects in both.
        zero_f1_subjects: Phase 1 subjects with F1 == 0.
        zero_f1_in_worst_quartile: Of those, how many fall in Phase 2's worst EER quartile.
            Descriptive only; the verdict rests on rho.
        expected_in_worst_quartile_by_chance: len(zero_f1_subjects) / 4.
        verdict: Under the CONCORDANCE_* constants.
    """

    spearman_rho: float
    p_value: float
    n_subjects: int
    zero_f1_subjects: tuple[int, ...]
    zero_f1_in_worst_quartile: int
    expected_in_worst_quartile_by_chance: float
    verdict: Literal["confirmed", "refuted", "inconclusive"]


def phase1_concordance(
    per_subject: Mapping[int, float],
    phase1_f1: Mapping[int, float],
    *,
    n_permutations: int = N_PERMUTATIONS,
    seed: int = PERMUTATION_SEED,
) -> ConcordanceResult:
    """Test P1b against the committed Phase 1 per-class F1.

    Evaluated once, on the headline table: cross-condition, protected domain, holdout
    impostors. Other tables do not get a verdict; fixing one table up front avoids a
    choice among forking paths. Caveat recorded with the result: both phases score the same
    recordings, so a subject with a poor eyes-closed recording can land in both tails
    without identity features being the reason.

    Raises:
        ValueError: If fewer than 10 subjects are in both mappings.
    """
    raise NotImplementedError("TODO(phase-2): Spearman + permutation test")


def bootstrap_eer_ci(
    table: ScoreTable,
    impostor_source: ImpostorSource,
    *,
    n_boot: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
    level: float = 0.95,
) -> tuple[float, float]:
    """Percentile CI for pooled EER by two-way subject bootstrap.

    Claimed subjects and impostor subjects are resampled independently, with replacement.
    Windows are never resampled individually, because that would treat 56 windows from one
    person as 56 people.
    """
    raise NotImplementedError("TODO(phase-2): two-way subject bootstrap")


# --------------------------------------------------------------- session-level


@dataclass(frozen=True)
class SessionTrace:
    """One replayed session, as seen by the metrics.

    Attributes:
        claimed_subject: Enrolled identity the session was opened for.
        decision_times_s: (n,) WindowScore.decision_time_s, ascending.
        states: Length n, the session state after each decision.
        swap_at_s: Session time at which the injected signal begins crossfading to another
            source, or None for a genuine-only replay.
        impostor_subject: Source after the swap. Equals claimed_subject for a self-splice
            control, and is None for genuine-only.
    """

    claimed_subject: int
    decision_times_s: NDArray[np.float64]
    states: tuple[SessionStateName, ...]
    swap_at_s: float | None
    impostor_subject: int | None


def time_to_detect(
    trace: SessionTrace,
    target_state: Literal["challenged", "revoked"],
) -> float | None:
    """Seconds from swap onset to the first decision at or after it that reaches target_state.

    Measured at decision time, which includes the window length and the right context
    margin, because that is the earliest moment the system could act. With the Phase 2
    replay constants, no decision reflects any post-swap signal before 3 s, and none rests
    entirely on impostor signal before 5 s. Those are floors set by the signal path, not
    expected values: session confidence accumulates over several windows, so measured
    times are longer, and the distribution is what gets reported. "Reaches" means
    states[i] is target_state or a later one in active < challenged < revoked. A transition
    before the swap is not a detection; false_transition_rate counts those.

    Returns:
        Seconds, or None if the target is never reached (censored).

    Raises:
        ValueError: If trace.swap_at_s is None.
    """
    raise NotImplementedError("TODO(phase-2): time to detect")


@dataclass(frozen=True)
class DetectionSummary:
    """Time-to-detect over impostor-swap replays. Never reported without FalseTransitionSummary.

    Attributes:
        target_state: "challenged" or "revoked".
        n_traces: Swap replays.
        n_censored: Not detected within the horizon.
        median_s: Median with censored traces as +inf, so it is inf if more than half are
            censored.
        p90_s: As median_s.
        fraction_detected_within_horizon: Detected within horizon_s of the swap.
        horizon_s: Seconds available after the swap.
    """

    target_state: str
    n_traces: int
    n_censored: int
    median_s: float
    p90_s: float
    fraction_detected_within_horizon: float
    horizon_s: float


def summarize_time_to_detect(
    traces: Sequence[SessionTrace],
    target_state: Literal["challenged", "revoked"],
    *,
    horizon_s: float,
) -> DetectionSummary:
    """Summarize detection over swap replays. Raises ValueError on an empty or swap-free input."""
    raise NotImplementedError("TODO(phase-2): detection summary")


@dataclass(frozen=True)
class FalseTransitionSummary:
    """How often genuine sessions are challenged or revoked. The price of fast detection.

    A detector that revokes every session at once has a time-to-detect of zero, so this is
    reported beside every DetectionSummary.

    Attributes:
        target_state: "challenged" or "revoked".
        n_sessions: Genuine-only (or self-splice) replays.
        fraction_of_sessions: Sessions that reached target_state at all.
        events_per_hour: Target transitions per hour of genuine exposure.
        exposure_s: Total scored session time.
    """

    target_state: str
    n_sessions: int
    fraction_of_sessions: float
    events_per_hour: float
    exposure_s: float


def false_transition_rate(
    traces: Sequence[SessionTrace],
    target_state: Literal["challenged", "revoked"],
) -> FalseTransitionSummary:
    """Rate of target_state on sessions where every sample is the claimed subject's.

    Raises:
        ValueError: If any trace contains another subject's signal.
    """
    raise NotImplementedError("TODO(phase-2): false challenge / revoke rate")


# --------------------------------------------------------------- report


@dataclass(frozen=True)
class VerificationReport:
    """Open-set verification results for one table and impostor source. No accuracy field.

    Attributes:
        split_kind: "cross_condition" or "temporal".
        domain: "protected" (headline) or "embedding".
        impostor_source: "impostor_holdout" (headline) or "cohort".
        eer: Oracle EER point on this table.
        headline: FRR at HEADLINE_FAR. The headline number, with its own under_resolved
            flag.
        frr_at_far: One OperatingPoint per FAR_TARGETS, each with its under_resolved flag.
        deployed: Rates on this table at the threshold selected on cohort scores for
            FAR = 0.01, or None when this table is the cohort table.
        eer_ci95: Two-way subject bootstrap, or None if not run.
        per_subject_eer: Claimed subject -> EER.
        tail: Distribution summary and the P1a verdict.
        random_pairing_control_eer: Must reach RANDOM_PAIRING_CONTROL_MIN_EER.
        n_genuine_scores: Genuine comparisons.
        n_impostor_scores: Impostor comparisons from this source.
        n_claimed_subjects: Enrolled subjects evaluated.
        n_impostor_subjects: Distinct impostor sources.
        n_impostor_pairs: Distinct (claimed, impostor) pairs.
        n_probe_not_ok: Probe windows flagged by the quality mask, scored not excluded.
        n_failure_to_enroll: Subjects whose enrollment was refused.
        model_versions: One per fold.
        streaming_fingerprint: Of every feature matrix involved.
    """

    split_kind: str
    domain: ScoreDomain
    impostor_source: ImpostorSource
    eer: OperatingPoint
    headline: OperatingPoint
    frr_at_far: tuple[OperatingPoint, ...]
    deployed: OperatingPoint | None
    eer_ci95: tuple[float, float] | None
    per_subject_eer: dict[int, float]
    tail: TailSummary
    random_pairing_control_eer: float
    n_genuine_scores: int
    n_impostor_scores: int
    n_claimed_subjects: int
    n_impostor_subjects: int
    n_impostor_pairs: int
    n_probe_not_ok: int
    n_failure_to_enroll: int
    model_versions: tuple[str, ...]
    streaming_fingerprint: str


def check_random_pairing_control(report: VerificationReport) -> None:
    """Refuse a report whose pairing control did not sit near chance.

    Raises:
        AssertionError: If random_pairing_control_eer < RANDOM_PAIRING_CONTROL_MIN_EER.
    """
    raise NotImplementedError("TODO(phase-2): pairing control gate")


def write_verification_report(report: VerificationReport, out_dir: Path, *, prefix: str) -> None:
    """Write {prefix}_metrics.json and {prefix}_per_subject_eer.csv. Scores and rates only.

    The per-subject CSV also carries each subject's genuine and impostor score counts and
    n_probe_not_ok, so the tail can be read alongside quality.
    """
    raise NotImplementedError("TODO(phase-2): report artifacts")
