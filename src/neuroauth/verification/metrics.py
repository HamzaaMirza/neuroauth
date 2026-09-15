"""Biometric error rates, the per-subject tail, and time-to-detect. No accuracy, anywhere.

Hard rule 2 and D-012: no report here has an accuracy field, and a test pins that.

Every constant below judges a result. They were proposed in the Phase 2 contracts and fixed
when this file was committed (0a4abb7), before any Phase 2 result existed (D-016 rule).
tests/test_metrics.py pins them.
"""

import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import numpy as np
from matplotlib import rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from numpy.typing import NDArray
from scipy.special import ndtri
from scipy.stats import rankdata

ScoreDomain = Literal["embedding", "protected", "decision"]
"""embedding: cosine similarity of unprotected embeddings, computed in memory only, to
measure the cost of protection. protected: Hamming similarity of BioHash bits, which is the
headline registered before D-021. decision: the v0 decision layer's log-likelihood ratio
(D-021), cross-fitted across folds."""

ImpostorSource = Literal["impostor_holdout", "cohort"]
"""impostor_holdout: the 20 committed subjects, used only for reporting. cohort: other
enrollable subjects in the same fold, never fitted on in that fold. Used for choosing
thresholds and fitting the decision layer, and never reported as the headline FAR."""

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

# DET plot styling: categorical slots 1-3 of the reference palette, which validate
# all-pairs (crossing lines put every pair in contact), on the light chart surface.
_SERIES: Final = ("#2a78d6", "#eb6834", "#1baf7a")
_SURFACE: Final = "#fcfcfb"
_INK_PRIMARY: Final = "#0b0b0b"
_INK_SECONDARY: Final = "#52514e"
_INK_MUTED: Final = "#898781"
_GRID: Final = "#e1e0d9"
_BASELINE: Final = "#c3c2b7"
_FONT: Final = "DejaVu Sans"


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
        template_score_mean: (n,) the claimed template's enrollment statistic mu (D-021).
            NaN in the embedding domain, which has no protected template.
        template_score_std: (n,) the claimed template's sigma, NaN as above.
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
    template_score_mean: NDArray[np.float64]
    template_score_std: NDArray[np.float64]
    domain: ScoreDomain
    split_kind: str


def _columns(table: ScoreTable) -> tuple[NDArray[np.generic], ...]:
    return (
        table.scores,
        table.claimed_subject,
        table.source_subject,
        table.source_is_holdout,
        table.fold,
        table.probe_onset_s,
        table.probe_window_ok,
        table.template_score_mean,
        table.template_score_std,
    )


def select_rows(table: ScoreTable, mask: NDArray[np.bool_]) -> ScoreTable:
    """The rows of table where mask is True, as a new table."""
    return ScoreTable(
        scores=table.scores[mask],
        claimed_subject=table.claimed_subject[mask],
        source_subject=table.source_subject[mask],
        source_is_holdout=table.source_is_holdout[mask],
        fold=table.fold[mask],
        probe_onset_s=table.probe_onset_s[mask],
        probe_window_ok=table.probe_window_ok[mask],
        template_score_mean=table.template_score_mean[mask],
        template_score_std=table.template_score_std[mask],
        domain=table.domain,
        split_kind=table.split_kind,
    )


def concatenate_tables(tables: Sequence[ScoreTable]) -> ScoreTable:
    """Stack tables of the same domain and split, e.g. one per fold.

    Raises:
        ValueError: If tables is empty or domains or splits differ.
    """
    if not tables:
        raise ValueError("no score tables to concatenate")
    first = tables[0]
    if any((t.domain, t.split_kind) != (first.domain, first.split_kind) for t in tables):
        raise ValueError("score tables differ in domain or split_kind")
    return ScoreTable(
        scores=np.concatenate([t.scores for t in tables]),
        claimed_subject=np.concatenate([t.claimed_subject for t in tables]),
        source_subject=np.concatenate([t.source_subject for t in tables]),
        source_is_holdout=np.concatenate([t.source_is_holdout for t in tables]),
        fold=np.concatenate([t.fold for t in tables]),
        probe_onset_s=np.concatenate([t.probe_onset_s for t in tables]),
        probe_window_ok=np.concatenate([t.probe_window_ok for t in tables]),
        template_score_mean=np.concatenate([t.template_score_mean for t in tables]),
        template_score_std=np.concatenate([t.template_score_std for t in tables]),
        domain=first.domain,
        split_kind=first.split_kind,
    )


def check_score_table(table: ScoreTable, impostor_holdout: frozenset[int]) -> None:
    """Structural and cohort integrity, run on every table before any metric.

    Raises:
        AssertionError: If any claimed_subject is in the holdout (never enrolled),
            source_is_holdout disagrees with the holdout set, or any score is non-finite.
        ValueError: If columns are not 1-D arrays of one length.
    """
    n_rows = table.scores.shape[0]
    if any(column.ndim != 1 or column.shape[0] != n_rows for column in _columns(table)):
        raise ValueError("score table columns must be 1-D arrays of one length")
    if not np.isfinite(table.scores).all():
        raise AssertionError("score table contains non-finite scores")
    claimed_holdout = sorted(set(table.claimed_subject.tolist()) & impostor_holdout)
    if claimed_holdout:
        raise AssertionError(
            f"impostor-holdout subjects appear as claimed identities: {claimed_holdout}"
        )
    expected = np.isin(table.source_subject, np.array(sorted(impostor_holdout), dtype=np.int64))
    if not np.array_equal(expected, table.source_is_holdout):
        raise AssertionError("source_is_holdout disagrees with the impostor holdout")


def genuine_mask(table: ScoreTable) -> NDArray[np.bool_]:
    mask: NDArray[np.bool_] = table.claimed_subject == table.source_subject
    return mask


def impostor_mask(table: ScoreTable, impostor_source: ImpostorSource) -> NDArray[np.bool_]:
    """Impostor rows from one source."""
    from_holdout = (
        table.source_is_holdout
        if impostor_source == "impostor_holdout"
        else ~table.source_is_holdout
    )
    mask: NDArray[np.bool_] = (table.claimed_subject != table.source_subject) & from_holdout
    return mask


def split_scores(
    table: ScoreTable,
    impostor_source: ImpostorSource,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """(genuine, impostor) scores, with impostor rows restricted to one source."""
    return table.scores[genuine_mask(table)], table.scores[impostor_mask(table, impostor_source)]


def n_impostor_pairs(table: ScoreTable, impostor_source: ImpostorSource) -> int:
    """Distinct (claimed, impostor subject) pairs behind the impostor scores of one source."""
    mask = impostor_mask(table, impostor_source)
    pairs = np.stack((table.claimed_subject[mask], table.source_subject[mask]), axis=1)
    return int(np.unique(pairs, axis=0).shape[0]) if pairs.size else 0


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


def _scores(values: NDArray[np.float64], side: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{side} scores must be a non-empty 1-D array, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{side} scores contain non-finite values")
    return array


def error_rates(genuine: NDArray[np.float64], impostor: NDArray[np.float64]) -> ErrorRates:
    """FAR and FRR at every distinct threshold, with accept iff score >= threshold.

    Tied scores are accepted or rejected together. No operating point between ties is
    invented, which matters because Hamming similarity moves in steps of 1/64.

    Raises:
        ValueError: If either side is empty, not 1-D, or non-finite. This is evaluation,
            not inference, so bad input raises.
    """
    g = _scores(genuine, "genuine")
    i = _scores(impostor, "impostor")
    thresholds = np.append(np.unique(np.concatenate((g, i))), np.inf)
    below_genuine = np.searchsorted(np.sort(g), thresholds, side="left")
    below_impostor = np.searchsorted(np.sort(i), thresholds, side="left")
    return ErrorRates(
        thresholds=thresholds,
        far=(i.size - below_impostor) / i.size,
        frr=below_genuine / g.size,
        n_genuine=int(g.size),
        n_impostor=int(i.size),
    )


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


def eer_value(point: OperatingPoint) -> float:
    """The EER of an equal_error_rate point: max(FAR, FRR)."""
    return max(point.far, point.frr)


def equal_error_rate(rates: ErrorRates) -> OperatingPoint:
    """EER as the smallest achievable max(FAR, FRR), with the threshold that achieves it.

    Chosen over interpolating the FAR/FRR crossing, which reports a rate that no single
    threshold attains, and over the ROC-convex-hull EER, which is harder to explain for no
    gain at these sample sizes. It can exceed the interpolated value by at most one step,
    max(1/n_genuine, 1/n_impostor). The returned far and frr are both reported, and EER is
    their maximum (eer_value). Ties between thresholds go to the lowest one.
    """
    index = int(np.argmin(np.maximum(rates.far, rates.frr)))
    return OperatingPoint(
        threshold=float(rates.thresholds[index]),
        far=float(rates.far[index]),
        frr=float(rates.frr[index]),
        target_far=None,
        under_resolved=None,
    )


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
    if not 0.0 < target_far < 1.0:
        raise ValueError(f"target_far must be in (0, 1), got {target_far}")
    if n_impostor_pairs < 1:
        raise ValueError(f"n_impostor_pairs must be at least 1, got {n_impostor_pairs}")
    index = int(np.flatnonzero(rates.far <= target_far)[0])
    return OperatingPoint(
        threshold=float(rates.thresholds[index]),
        far=float(rates.far[index]),
        frr=float(rates.frr[index]),
        target_far=target_far,
        under_resolved=n_impostor_pairs * target_far < MIN_EXPECTED_ERRORS,
    )


def rates_at_threshold(
    genuine: NDArray[np.float64],
    impostor: NDArray[np.float64],
    threshold: float,
) -> OperatingPoint:
    """FAR and FRR at a threshold chosen elsewhere.

    This is how a threshold selected on cohort scores is reported on the holdout. It is the
    deployable number, as opposed to the oracle EER.
    """
    g = _scores(genuine, "genuine")
    i = _scores(impostor, "impostor")
    return OperatingPoint(
        threshold=float(threshold),
        far=float(np.mean(i >= threshold)),
        frr=float(np.mean(g < threshold)),
        target_far=None,
        under_resolved=None,
    )


def _probit(rate: NDArray[np.float64] | float, n: int) -> NDArray[np.float64]:
    clipped = np.clip(np.asarray(rate, dtype=np.float64), 0.5 / n, 1.0 - 0.5 / n)
    probit: NDArray[np.float64] = ndtri(clipped)
    return probit


def det_coordinates(rates: ErrorRates) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Probit-transformed (FAR, FRR) for a DET plot.

    Rates of exactly 0 or 1 are clipped to 0.5/n and 1 - 0.5/n of their side before the
    inverse normal CDF, so a curve reaching zero errors is drawn at the edge of what its
    sample size can show.
    """
    return _probit(rates.far, rates.n_impostor), _probit(rates.frr, rates.n_genuine)


def write_det_plot(
    curves: Mapping[str, ErrorRates],
    path: Path,
    *,
    title: str,
    subtitle: str,
    eer_points: Mapping[str, OperatingPoint],
    marked_far: float = HEADLINE_FAR,
) -> None:
    """Write a DET plot, one curve per label, with EER markers and the headline FAR marked.

    Axes are labelled in rates (0.1%, 1%, 5%, 20%, 50%) on probit scales. At most three
    curves: the palette's first three slots are the ones validated for every pair, and DET
    curves cross. Identity is carried by the legend, which also states each EER, never by
    color alone; the metrics JSON is the table view.

    Raises:
        ValueError: If there are no curves or more than three.
    """
    if not curves or len(curves) > len(_SERIES):
        raise ValueError(f"a DET plot takes 1 to {len(_SERIES)} curves, got {len(curves)}")
    rate_ticks = np.array([0.001, 0.01, 0.05, 0.2, 0.5])
    tick_labels = ["0.1%", "1%", "5%", "20%", "50%"]
    low, high = float(ndtri(0.0005)), float(ndtri(0.7))

    with rc_context({"font.family": _FONT}):
        figure = Figure(figsize=(6.4, 6.4), dpi=150, facecolor=_SURFACE)
        FigureCanvasAgg(figure)
        axes = figure.add_subplot(facecolor=_SURFACE)
        ticks = ndtri(rate_ticks)
        axes.set_xticks(ticks, tick_labels)
        axes.set_yticks(ticks, tick_labels)
        axes.set_xlim(low, high)
        axes.set_ylim(low, high)
        axes.grid(True, color=_GRID, linewidth=0.5)
        axes.set_axisbelow(True)
        axes.tick_params(colors=_INK_MUTED, labelsize=8, length=0)
        for spine in axes.spines.values():
            spine.set_visible(False)
        axes.plot([low, high], [low, high], color=_BASELINE, linewidth=0.5, zorder=1)
        marked = float(ndtri(marked_far))
        axes.axvline(marked, color=_BASELINE, linewidth=0.5, zorder=1)
        axes.text(marked, high, f" FAR {marked_far:.0%}", color=_INK_MUTED, fontsize=7, va="top")

        handles: list[Line2D] = []
        for (label, rates), color in zip(curves.items(), _SERIES, strict=False):
            x, y = det_coordinates(rates)
            axes.plot(
                x,
                y,
                color=color,
                linewidth=1.0,
                solid_joinstyle="round",
                solid_capstyle="round",
                zorder=3,
            )
            legend_label = label
            point = eer_points.get(label)
            if point is not None:
                axes.scatter(
                    _probit(point.far, rates.n_impostor),
                    _probit(point.frr, rates.n_genuine),
                    s=36,
                    color=color,
                    edgecolors=_SURFACE,
                    linewidths=1.0,
                    zorder=4,
                )
                legend_label = f"{label}   EER {eer_value(point):.1%}"
            handles.append(Line2D([0], [0], color=color, linewidth=1.5, label=legend_label))
        axes.legend(
            handles=handles,
            loc="upper right",
            frameon=False,
            fontsize=8,
            labelcolor=_INK_SECONDARY,
        )
        axes.set_xlabel("False accept rate", color=_INK_SECONDARY, fontsize=9)
        axes.set_ylabel("False reject rate", color=_INK_SECONDARY, fontsize=9)
        figure.subplots_adjust(left=0.12, right=0.96, bottom=0.09, top=0.87)
        figure.text(
            0.12, 0.965, title, color=_INK_PRIMARY, fontsize=11, fontweight="bold", va="top"
        )
        figure.text(0.12, 0.925, subtitle, color=_INK_SECONDARY, fontsize=8, va="top")
        figure.savefig(path, facecolor=_SURFACE)


# --------------------------------------------------------------- per-subject tail


@dataclass(frozen=True)
class PerSubjectRow:
    """One claimed subject's verification result.

    Attributes:
        subject_id: Claimed subject.
        eer: Per-subject EER (per-subject threshold, so optimistic).
        n_genuine: Genuine comparisons.
        n_impostor: Impostor comparisons from the report's source.
        n_genuine_not_ok: Genuine probe windows flagged by the quality mask.
    """

    subject_id: int
    eer: float
    n_genuine: int
    n_impostor: int
    n_genuine_not_ok: int


def per_subject_rows(
    table: ScoreTable, impostor_source: ImpostorSource
) -> tuple[PerSubjectRow, ...]:
    """Per-subject EER with counts, for every claimed subject, in ascending id order.

    Raises:
        ValueError: If a claimed subject has no genuine or no impostor scores.
    """
    genuine = genuine_mask(table)
    impostor = impostor_mask(table, impostor_source)
    rows: list[PerSubjectRow] = []
    for subject in np.unique(table.claimed_subject).tolist():
        own = table.claimed_subject == subject
        g = table.scores[genuine & own]
        i = table.scores[impostor & own]
        if g.size == 0 or i.size == 0:
            raise ValueError(f"subject {subject} has {g.size} genuine and {i.size} impostor scores")
        rows.append(
            PerSubjectRow(
                subject_id=int(subject),
                eer=eer_value(equal_error_rate(error_rates(g, i))),
                n_genuine=int(g.size),
                n_impostor=int(i.size),
                n_genuine_not_ok=int((~table.probe_window_ok[genuine & own]).sum()),
            )
        )
    return tuple(rows)


def per_subject_eer(table: ScoreTable, impostor_source: ImpostorSource) -> dict[int, float]:
    """EER per claimed subject, from its genuine scores and its impostor scores.

    Uses a per-subject threshold, which makes it optimistic compared with the pooled EER's
    single global threshold. Both are reported and never conflated.

    Returns:
        Claimed subject -> EER, for every claimed subject in the table.
    """
    return {row.subject_id: row.eer for row in per_subject_rows(table, impostor_source)}


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


def _worst_first(values: Mapping[int, float]) -> list[int]:
    return [subject for subject, _ in sorted(values.items(), key=lambda item: (-item[1], item[0]))]


def summarize_tail(per_subject: Mapping[int, float], *, pooled_eer: float) -> TailSummary:
    """Summarize per-subject EER and apply the a priori heavy-tail rule.

    Raises:
        ValueError: If per_subject is empty.
    """
    if not per_subject:
        raise ValueError("no per-subject EERs to summarize")
    values = np.array(list(per_subject.values()), dtype=np.float64)
    n_worst = math.ceil(TAIL_WORST_FRACTION * values.size)
    worst = tuple(_worst_first(per_subject)[:n_worst])
    median = float(np.median(values))
    worst_mean = float(np.mean([per_subject[subject] for subject in worst]))
    return TailSummary(
        n_subjects=int(values.size),
        median=median,
        p90=float(np.percentile(values, 90)),
        worst_decile_mean=worst_mean,
        worst_decile_subjects=worst,
        pooled_eer=float(pooled_eer),
        heavy_tail=worst_mean >= TAIL_HEAVY_MIN_RATIO * median
        and worst_mean >= median + TAIL_HEAVY_MIN_GAP,
    )


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
        ValueError: If fewer than 10 subjects are in both mappings, or either variable is
            constant over them.
    """
    common = sorted(set(per_subject) & set(phase1_f1))
    if len(common) < 10:
        raise ValueError(f"need at least 10 subjects in both mappings, got {len(common)}")
    f1_ranks = rankdata([phase1_f1[subject] for subject in common])
    eer_ranks = rankdata([per_subject[subject] for subject in common])
    centered_f1 = f1_ranks - f1_ranks.mean()
    centered_eer = eer_ranks - eer_ranks.mean()
    norm = float(np.linalg.norm(centered_f1) * np.linalg.norm(centered_eer))
    if norm == 0.0:
        raise ValueError("Phase 1 F1 or Phase 2 EER is constant over the common subjects")
    rho = float(centered_f1 @ centered_eer) / norm

    rng = np.random.default_rng(seed)
    permuted = np.stack([rng.permutation(centered_eer) for _ in range(n_permutations)])
    null = (permuted @ centered_f1) / norm
    p_value = float((1 + np.count_nonzero(null <= rho)) / (1 + n_permutations))

    zero = tuple(subject for subject in common if phase1_f1[subject] == 0.0)
    quartile = set(_worst_first({s: per_subject[s] for s in common})[: math.ceil(len(common) / 4)])
    if rho <= CONCORDANCE_CONFIRM_MAX_RHO and p_value < CONCORDANCE_ALPHA:
        verdict: Literal["confirmed", "refuted", "inconclusive"] = "confirmed"
    elif rho > CONCORDANCE_REFUTE_MIN_RHO:
        verdict = "refuted"
    else:
        verdict = "inconclusive"
    return ConcordanceResult(
        spearman_rho=rho,
        p_value=p_value,
        n_subjects=len(common),
        zero_f1_subjects=zero,
        zero_f1_in_worst_quartile=sum(1 for subject in zero if subject in quartile),
        expected_in_worst_quartile_by_chance=len(zero) / 4.0,
        verdict=verdict,
    )


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
    person as 56 people. Implemented with multiplicity weights over one sorted score array,
    so each replicate costs a cumulative sum rather than a re-sort.

    Raises:
        ValueError: If level is outside (0, 1), n_boot is below 1, or a side has no rows.
    """
    if not 0.0 < level < 1.0 or n_boot < 1:
        raise ValueError("level must be in (0, 1) and n_boot at least 1")
    genuine = genuine_mask(table)
    impostor = impostor_mask(table, impostor_source)
    if not genuine.any() or not impostor.any():
        raise ValueError("bootstrap needs genuine and impostor rows")
    used = genuine | impostor
    order = np.argsort(table.scores[used], kind="stable")
    scores = table.scores[used][order]
    is_genuine = genuine[used][order]
    claimed = table.claimed_subject[used][order]
    source = table.source_subject[used][order]

    claimed_ids = np.unique(table.claimed_subject[genuine])
    impostor_ids = np.unique(table.source_subject[impostor])
    claimed_index = np.searchsorted(claimed_ids, claimed)
    impostor_index = np.minimum(np.searchsorted(impostor_ids, source), impostor_ids.size - 1)
    first_of_value = np.flatnonzero(np.r_[True, scores[1:] != scores[:-1]])

    rng = np.random.default_rng(seed)
    replicates = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        claimed_counts = np.bincount(
            rng.integers(0, claimed_ids.size, claimed_ids.size), minlength=claimed_ids.size
        )
        impostor_counts = np.bincount(
            rng.integers(0, impostor_ids.size, impostor_ids.size), minlength=impostor_ids.size
        )
        w_claimed = claimed_counts[claimed_index].astype(np.float64)
        w_genuine = np.where(is_genuine, w_claimed, 0.0)
        w_impostor = np.where(is_genuine, 0.0, w_claimed * impostor_counts[impostor_index])
        total_genuine, total_impostor = w_genuine.sum(), w_impostor.sum()
        if total_genuine == 0.0 or total_impostor == 0.0:
            replicates[b] = np.nan
            continue
        cum_genuine = np.concatenate(([0.0], np.cumsum(w_genuine)))
        cum_impostor = np.concatenate(([0.0], np.cumsum(w_impostor)))
        frr = cum_genuine[first_of_value] / total_genuine
        far = (total_impostor - cum_impostor[first_of_value]) / total_impostor
        replicates[b] = min(float(np.min(np.maximum(far, frr))), 1.0)
    tail = (1.0 - level) / 2.0 * 100.0
    low, high = np.nanpercentile(replicates, [tail, 100.0 - tail])
    return float(low), float(high)


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


_STATE_RANK: Final[dict[str, int]] = {"active": 0, "challenged": 1, "revoked": 2}


def _rank(state: str) -> int:
    return _STATE_RANK.get(state, -1)


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
    times are longer, and the distribution is what gets reported.

    "Reaches" means states[i] is target_state or a later one in active < challenged <
    revoked. A session that reached the target before the swap has no detection to time and
    returns None; false_transition_rate counts those transitions on genuine replays.

    Returns:
        Seconds, or None if the target is never reached after the swap (censored).

    Raises:
        ValueError: If trace.swap_at_s is None, or states and decision times differ in length.
    """
    if trace.swap_at_s is None:
        raise ValueError("time_to_detect needs a trace with a swap")
    times = np.asarray(trace.decision_times_s, dtype=np.float64)
    if times.shape != (len(trace.states),):
        raise ValueError("decision_times_s and states differ in length")
    target = _rank(target_state)
    for time_s, state in zip(times.tolist(), trace.states, strict=True):
        if _rank(state) >= target:
            return None if time_s < trace.swap_at_s else time_s - trace.swap_at_s
    return None


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
    if not traces:
        raise ValueError("no traces to summarize")
    values = []
    for trace in traces:
        detected = time_to_detect(trace, target_state)
        values.append(detected if detected is not None and detected <= horizon_s else np.inf)
    times = np.array(values, dtype=np.float64)
    n_censored = int(np.isinf(times).sum())
    return DetectionSummary(
        target_state=target_state,
        n_traces=len(traces),
        n_censored=n_censored,
        median_s=float(np.quantile(times, 0.5, method="inverted_cdf")),
        p90_s=float(np.quantile(times, 0.9, method="inverted_cdf")),
        fraction_detected_within_horizon=1.0 - n_censored / len(traces),
        horizon_s=horizon_s,
    )


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
        ValueError: If traces is empty or any trace contains another subject's signal.
    """
    if not traces:
        raise ValueError("no traces to summarize")
    target = _rank(target_state)
    events = 0
    reached = 0
    exposure = 0.0
    for trace in traces:
        if trace.swap_at_s is not None and trace.impostor_subject != trace.claimed_subject:
            raise ValueError(f"trace for subject {trace.claimed_subject} contains another subject")
        ranks = [_rank(state) for state in trace.states]
        entered = sum(
            1
            for i, rank in enumerate(ranks)
            if rank >= target and (i == 0 or ranks[i - 1] < target)
        )
        events += entered
        reached += int(entered > 0)
        if len(trace.decision_times_s) > 1:
            exposure += float(trace.decision_times_s[-1] - trace.decision_times_s[0])
    return FalseTransitionSummary(
        target_state=target_state,
        n_sessions=len(traces),
        fraction_of_sessions=reached / len(traces),
        events_per_hour=events / exposure * 3600.0 if exposure > 0.0 else 0.0,
        exposure_s=exposure,
    )


# --------------------------------------------------------------- report


@dataclass(frozen=True)
class VerificationReport:
    """Open-set verification results for one table and impostor source. No accuracy field.

    Attributes:
        split_kind: "cross_condition" or "temporal".
        domain: "protected" (headline), "embedding", or "decision".
        impostor_source: "impostor_holdout" (headline) or "cohort".
        eer: Oracle EER point on this table.
        headline: FRR at HEADLINE_FAR. The headline number, with its own under_resolved
            flag.
        frr_at_far: One OperatingPoint per FAR_TARGETS, each with its under_resolved flag.
        deployed: Rates on this table at the threshold selected on cohort scores for
            FAR = 0.01, or None when no threshold was supplied.
        eer_ci95: Two-way subject bootstrap, or None if not run.
        per_subject: One row per claimed subject.
        tail: Distribution summary and the P1a verdict.
        random_pairing_control_eer: Must reach RANDOM_PAIRING_CONTROL_MIN_EER.
        n_genuine_scores: Genuine comparisons.
        n_impostor_scores: Impostor comparisons from this source.
        n_claimed_subjects: Enrolled subjects evaluated.
        n_impostor_subjects: Distinct impostor sources.
        n_impostor_pairs: Distinct (claimed, impostor) pairs.
        n_probe_not_ok: Distinct probe windows flagged by the quality mask, scored not
            excluded.
        n_failure_to_enroll: Subjects whose enrollment was refused.
        representation_versions: One per fold (D-021).
        decision_versions: One per fold for the decision domain, else empty.
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
    per_subject: tuple[PerSubjectRow, ...]
    tail: TailSummary
    random_pairing_control_eer: float
    n_genuine_scores: int
    n_impostor_scores: int
    n_claimed_subjects: int
    n_impostor_subjects: int
    n_impostor_pairs: int
    n_probe_not_ok: int
    n_failure_to_enroll: int
    representation_versions: tuple[str, ...]
    decision_versions: tuple[str, ...]
    streaming_fingerprint: str


def evaluate_verification(
    table: ScoreTable,
    *,
    impostor_holdout: frozenset[int],
    impostor_source: ImpostorSource,
    deployed_threshold: float | None,
    random_pairing_control_eer: float,
    n_failure_to_enroll: int,
    representation_versions: tuple[str, ...],
    streaming_fingerprint: str,
    decision_versions: tuple[str, ...] = (),
    bootstrap: bool = True,
) -> VerificationReport:
    """Every Phase 2 verification metric for one table and one impostor source.

    Raises:
        AssertionError: If check_score_table fails.
        ValueError: If a side has no scores or a claimed subject lacks genuine or impostor
            rows.
    """
    check_score_table(table, impostor_holdout)
    genuine, impostor = split_scores(table, impostor_source)
    rates = error_rates(genuine, impostor)
    pairs = n_impostor_pairs(table, impostor_source)
    eer = equal_error_rate(rates)
    used = genuine_mask(table) | impostor_mask(table, impostor_source)
    not_ok = ~table.probe_window_ok & used
    probe_windows_not_ok = np.unique(
        np.stack((table.source_subject[not_ok], table.probe_onset_s[not_ok]), axis=1), axis=0
    )
    rows = per_subject_rows(table, impostor_source)
    return VerificationReport(
        split_kind=table.split_kind,
        domain=table.domain,
        impostor_source=impostor_source,
        eer=eer,
        headline=frr_at_far(rates, HEADLINE_FAR, n_impostor_pairs=pairs),
        frr_at_far=tuple(frr_at_far(rates, t, n_impostor_pairs=pairs) for t in FAR_TARGETS),
        deployed=(
            None
            if deployed_threshold is None
            else rates_at_threshold(genuine, impostor, deployed_threshold)
        ),
        eer_ci95=bootstrap_eer_ci(table, impostor_source) if bootstrap else None,
        per_subject=rows,
        tail=summarize_tail({row.subject_id: row.eer for row in rows}, pooled_eer=eer_value(eer)),
        random_pairing_control_eer=random_pairing_control_eer,
        n_genuine_scores=int(genuine.size),
        n_impostor_scores=int(impostor.size),
        n_claimed_subjects=len(rows),
        n_impostor_subjects=int(
            np.unique(table.source_subject[impostor_mask(table, impostor_source)]).size
        ),
        n_impostor_pairs=pairs,
        n_probe_not_ok=int(probe_windows_not_ok.shape[0]),
        n_failure_to_enroll=n_failure_to_enroll,
        representation_versions=representation_versions,
        decision_versions=decision_versions,
        streaming_fingerprint=streaming_fingerprint,
    )


def check_random_pairing_control(report: VerificationReport) -> None:
    """Refuse a report whose pairing control did not sit near chance.

    Raises:
        AssertionError: If random_pairing_control_eer < RANDOM_PAIRING_CONTROL_MIN_EER.
    """
    if report.random_pairing_control_eer < RANDOM_PAIRING_CONTROL_MIN_EER:
        raise AssertionError(
            f"random-pairing control EER {report.random_pairing_control_eer:.3f} is below "
            f"{RANDOM_PAIRING_CONTROL_MIN_EER} on the {report.split_kind} {report.domain} table: "
            "'genuine' is picking up something other than same identity"
        )


def operating_point_json(point: OperatingPoint) -> dict[str, float | str | bool | None]:
    """An OperatingPoint as JSON-safe values; an infinite threshold is written as "inf"."""
    return {
        "threshold": point.threshold if math.isfinite(point.threshold) else "inf",
        "far": point.far,
        "frr": point.frr,
        "target_far": point.target_far,
        "under_resolved": point.under_resolved,
    }


def write_verification_report(report: VerificationReport, out_dir: Path, *, prefix: str) -> None:
    """Write {prefix}_metrics.json and {prefix}_per_subject_eer.csv. Scores and rates only.

    The per-subject CSV also carries each subject's genuine and impostor score counts and
    genuine probe windows flagged not-ok, so the tail can be read alongside quality.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tail = report.tail
    metrics = {
        "split_kind": report.split_kind,
        "domain": report.domain,
        "impostor_source": report.impostor_source,
        "headline_frr_at_far": operating_point_json(report.headline),
        "eer": eer_value(report.eer),
        "eer_point": operating_point_json(report.eer),
        "eer_ci95": list(report.eer_ci95) if report.eer_ci95 is not None else None,
        "frr_at_far": [operating_point_json(point) for point in report.frr_at_far],
        "deployed_at_cohort_threshold": (
            operating_point_json(report.deployed) if report.deployed is not None else None
        ),
        "tail": {
            "n_subjects": tail.n_subjects,
            "median": tail.median,
            "p90": tail.p90,
            "worst_decile_mean": tail.worst_decile_mean,
            "worst_decile_subjects": list(tail.worst_decile_subjects),
            "pooled_eer": tail.pooled_eer,
            "heavy_tail": tail.heavy_tail,
        },
        "random_pairing_control_eer": report.random_pairing_control_eer,
        "n_genuine_scores": report.n_genuine_scores,
        "n_impostor_scores": report.n_impostor_scores,
        "n_claimed_subjects": report.n_claimed_subjects,
        "n_impostor_subjects": report.n_impostor_subjects,
        "n_impostor_pairs": report.n_impostor_pairs,
        "n_probe_not_ok": report.n_probe_not_ok,
        "n_failure_to_enroll": report.n_failure_to_enroll,
        "representation_versions": list(report.representation_versions),
        "decision_versions": list(report.decision_versions),
        "streaming_fingerprint": report.streaming_fingerprint,
    }
    (out_dir / f"{prefix}_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    with (out_dir / f"{prefix}_per_subject_eer.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["subject_id", "eer", "n_genuine", "n_impostor", "n_genuine_not_ok"])
        for row in report.per_subject:
            writer.writerow(
                [
                    row.subject_id,
                    f"{row.eer:.6f}",
                    row.n_genuine,
                    row.n_impostor,
                    row.n_genuine_not_ok,
                ]
            )
