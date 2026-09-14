"""Phase 1 identification metrics, the shuffled-label gate, and the committed artifacts.

IdentificationReport deliberately has no `accuracy` field, and none will be added
(CLAUDE.md hard rule 2). The closed-set framing of Phase 1 is a scaffold for the
signal pipeline; Phase 2 replaces these metrics with EER, FAR, and FRR.

Nothing written by this module contains a feature vector or a raw sample (hard rule
3): only scores, counts, and quality-flag rates.
"""

import csv
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib import rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix, f1_score

from neuroauth.config import FRONTAL_EOG_CHANNELS
from neuroauth.dsp.features import feature_channels
from neuroauth.dsp.types import FeatureMatrix

CONTROL_MAX_CHANCE_RATIO = 3.0
"""Shuffled-label macro-F1 above this multiple of chance fails the control (D-007).

Set a priori, before any model was fit on real EEG. Not to be adjusted after seeing
results (D-016)."""

MATERIAL_DELTA = 0.05
"""Macro-F1 gap between absolute and relative band power treated as material (D-004).

Set a priori, before any model was fit on real EEG. Not to be adjusted after seeing
results (D-016)."""

ABLATION_MATERIAL_DROP = 0.05
"""Macro-F1 drop from removing a feature set that is treated as material (D-018).

Set a priori, after the first full run and before any ablation was run. Not to be
adjusted after seeing results (D-016)."""

ABLATION_SCOPE = (
    "The bound covers only information unique to the removed features. Muscle activity "
    "that remains in the retained features, and anything the model recovers from "
    "features correlated with the removed ones, is not bounded by this check."
)

QUALITY_REPORT_FILENAME = "quality_flag_rates.csv"

# Confusion-matrix styling: one-hue sequential ramp (blue, steps 100-700) on the
# light chart surface. Zero cells take the surface color so they recede.
_SEQUENTIAL_BLUE = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
_SURFACE = "#fcfcfb"
_INK_PRIMARY = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_INK_MUTED = "#898781"
_FONT = "DejaVu Sans"


def chance_level(n_classes: int) -> float:
    """Expected macro-F1 of uninformed guessing over n_classes balanced classes.

    Raises:
        ValueError: If n_classes is below 1.
    """
    if n_classes < 1:
        raise ValueError(f"n_classes must be at least 1, got {n_classes}")
    return 1.0 / n_classes


def macro_f1(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
    labels: Sequence[int],
) -> float:
    """Unweighted mean F1 over `labels`. A label with no support and no predictions
    scores 0, so a class the model never gets right is not silently dropped."""
    return float(f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0))


@dataclass(frozen=True)
class IdentificationReport:
    """Closed-set identification results for the Phase 1 baseline.

    Attributes:
        macro_f1: Unweighted mean of per-class F1.
        per_class_f1: (n_classes,) aligned with `labels`.
        confusion: (n_classes, n_classes) int, rows true, columns predicted.
        labels: Subject ids, in confusion-matrix row and column order.
        chance_level: 1 / n_classes. Reported alongside macro_f1 so the number has
            context -- 0.011 for 89 classes, and a reader cannot judge a macro-F1
            without it.
        shuffled_label_macro_f1: Macro-F1 of the same pipeline trained on shuffled
            labels. Must stay within CONTROL_MAX_CHANCE_RATIO of chance_level. None
            only when the control was not run, which check_shuffled_label_control
            refuses.
        n_train: Training rows used.
        n_test: Test rows scored.
        n_test_not_ok: Test rows the quality mask marked not-ok. In Phase 1 these
            are scored, not excluded, and the count is reported next to the score
            (D-015).
        split_kind: "temporal" or "cross_condition".
        normalization: "relative" or "absolute_log". Both are run; the relative
            result is the headline.
        n_subjects_enrollable: Subjects in the enrollable cohort.
        n_subjects_impostor_holdout: Subjects reserved as impostors and untouched by
            this evaluation.
        config_fingerprint: PipelineConfig.fingerprint() of the features.
    """

    macro_f1: float
    per_class_f1: NDArray[np.float64]
    confusion: NDArray[np.int64]
    labels: tuple[int, ...]
    chance_level: float
    shuffled_label_macro_f1: float | None
    n_train: int
    n_test: int
    n_test_not_ok: int
    split_kind: str
    normalization: str
    n_subjects_enrollable: int
    n_subjects_impostor_holdout: int
    config_fingerprint: str


def evaluate_identification(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
    labels: tuple[int, ...],
    *,
    split_kind: str,
    normalization: str,
    config_fingerprint: str,
    n_train: int,
    n_test_not_ok: int,
    n_subjects_enrollable: int,
    n_subjects_impostor_holdout: int,
    shuffled_label_macro_f1: float | None = None,
) -> IdentificationReport:
    """Compute macro-F1, per-class F1, the confusion matrix, and the chance level.

    Args:
        y_true: (n_test,) subject ids.
        y_pred: (n_test,) predicted subject ids.
        labels: Full label set, so classes absent from y_true still appear in the
            confusion matrix.
        split_kind: Recorded on the report for provenance.
        normalization: Recorded on the report for provenance.
        config_fingerprint: Recorded on the report for provenance.
        n_train: Recorded on the report.
        n_test_not_ok: Recorded on the report.
        n_subjects_enrollable: Recorded on the report.
        n_subjects_impostor_holdout: Recorded on the report.
        shuffled_label_macro_f1: Result of run_shuffled_label_control.

    Returns:
        An IdentificationReport.

    Raises:
        ValueError: If y_true and y_pred are not matching non-empty 1-D arrays,
            labels is empty, or either array holds an id outside labels -- which
            sklearn would otherwise drop from the confusion matrix without a word.
    """
    truth = np.asarray(y_true, dtype=np.int64)
    predicted = np.asarray(y_pred, dtype=np.int64)
    if truth.ndim != 1 or truth.shape != predicted.shape:
        raise ValueError(
            f"y_true and y_pred must be matching 1-D arrays, got {truth.shape} and "
            f"{predicted.shape}"
        )
    if truth.size == 0:
        raise ValueError("no test rows to evaluate")
    if not labels:
        raise ValueError("labels is empty")
    label_list = list(labels)
    outside = sorted(set(np.concatenate((truth, predicted)).tolist()) - set(label_list))
    if outside:
        raise ValueError(f"ids outside the label set: {outside}")

    per_class = np.asarray(
        f1_score(truth, predicted, labels=label_list, average=None, zero_division=0),
        dtype=np.float64,
    )
    return IdentificationReport(
        macro_f1=float(per_class.mean()),
        per_class_f1=per_class,
        confusion=np.asarray(confusion_matrix(truth, predicted, labels=label_list), dtype=np.int64),
        labels=tuple(labels),
        chance_level=chance_level(len(labels)),
        shuffled_label_macro_f1=shuffled_label_macro_f1,
        n_train=n_train,
        n_test=int(truth.size),
        n_test_not_ok=n_test_not_ok,
        split_kind=split_kind,
        normalization=normalization,
        n_subjects_enrollable=n_subjects_enrollable,
        n_subjects_impostor_holdout=n_subjects_impostor_holdout,
        config_fingerprint=config_fingerprint,
    )


def check_shuffled_label_control(
    report: IdentificationReport,
    *,
    max_ratio: float = CONTROL_MAX_CHANCE_RATIO,
) -> None:
    """Refuse a report whose shuffled-label control did not collapse to chance.

    Called before any artifact is written. It gates label leakage outside the train
    association; it cannot detect window-overlap leakage, which assert_no_window_overlap
    rules out structurally (D-007).

    Args:
        report: The report to check.
        max_ratio: Shuffled macro-F1 may be at most this multiple of chance.

    Raises:
        AssertionError: If the control was not run, or exceeds the ceiling.
    """
    shuffled = report.shuffled_label_macro_f1
    if shuffled is None:
        raise AssertionError("shuffled-label control was not run; refusing the report")
    ceiling = max_ratio * report.chance_level
    if shuffled > ceiling:
        raise AssertionError(
            f"shuffled-label macro-F1 {shuffled:.4f} exceeds {max_ratio:g}x chance "
            f"({ceiling:.4f}) on the {report.split_kind} split; labels are reaching the "
            "predictions by a path other than training"
        )


def _check_comparable(first: IdentificationReport, second: IdentificationReport) -> None:
    if (first.split_kind, first.labels, first.n_test) != (
        second.split_kind,
        second.labels,
        second.n_test,
    ):
        raise ValueError("reports differ in split, labels, or test rows; they are not comparable")


def compare_normalizations(
    relative: IdentificationReport,
    absolute: IdentificationReport,
    *,
    material_delta: float = MATERIAL_DELTA,
) -> dict[str, float | str]:
    """Summarize the relative-vs-absolute band-power comparison.

    The gap between the two is a finding, not a footnote -- but a bounded one. Absolute
    power carries amplitude from two sources that single-session data cannot separate:
    recording artifacts (electrode impedance, cap placement, amplifier gain), perfectly
    confounded with subject in this dataset, and anatomy (skull thickness, tissue
    conductivity), which is person-specific and would survive a second session. A
    material gap is therefore reported as an upper bound on the session-artifact
    contribution, never as a measure of it (D-004).

    Args:
        relative: Report from the "relative" run. The headline.
        absolute: Report from the "absolute_log" run.
        material_delta: Macro-F1 difference treated as material.

    Returns:
        A JSON-serializable summary: both macro-F1 values, the delta (absolute minus
        relative), the threshold used, and a plain-language interpretation for the
        README.

    Raises:
        ValueError: If the reports are passed in the wrong order, or were not scored
            on the same split, labels, and test rows.
    """
    if relative.normalization != "relative" or absolute.normalization != "absolute_log":
        raise ValueError(
            f"expected (relative, absolute_log) reports, got ({relative.normalization}, "
            f"{absolute.normalization})"
        )
    _check_comparable(relative, absolute)

    split = relative.split_kind.replace("_", "-")
    delta = absolute.macro_f1 - relative.macro_f1
    if delta > material_delta:
        interpretation = (
            f"Absolute band power scores {delta:.3f} macro-F1 higher than relative on the "
            f"{split} split. eegmmidb is single-session, so the amplitude information behind "
            "this gap cannot be separated from session artifacts (impedance, cap placement, "
            "amplifier gain): the gap is an upper bound on their contribution. Part of it "
            "may be anatomy, such as skull thickness, which is person-specific and would "
            "survive a second session."
        )
    elif delta < -material_delta:
        interpretation = (
            f"Relative band power scores {-delta:.3f} macro-F1 higher than absolute on "
            f"the {split} split; amplitude offsets are not inflating the score."
        )
    else:
        interpretation = (
            f"Relative and absolute band power score within {material_delta} macro-F1 "
            f"of each other on the {split} split; no evidence that amplitude offsets "
            "inflate the score."
        )
    return {
        "split_kind": relative.split_kind,
        "relative_macro_f1": relative.macro_f1,
        "absolute_log_macro_f1": absolute.macro_f1,
        "delta_absolute_minus_relative": delta,
        "material_delta": material_delta,
        "interpretation": interpretation,
    }


def summarize_ablation(
    headline: IdentificationReport,
    ablated: IdentificationReport,
    *,
    removed: str,
    material_drop: float = ABLATION_MATERIAL_DROP,
) -> dict[str, float | str | bool]:
    """Summarize how much the headline depends on a removed set of features.

    The drop bounds the contribution of information unique to the removed features; it
    does not decompose it. For the EMG check the removed features carry both muscle and
    neural activity, and scalp EEG cannot separate the two, so a drop -- material or
    not -- says nothing about how much of that information is EMG (D-018).

    Args:
        headline: The full-feature report.
        ablated: Scored on the same split, labels, and test rows without the removed
            features.
        removed: Plain-language description of what was removed, used in the text.
        material_drop: Macro-F1 drop treated as material.

    Returns:
        A JSON-serializable summary: both macro-F1 values, the drop (headline minus
        ablated), whether it is material, the threshold, an interpretation, and the
        scope of the bound.

    Raises:
        ValueError: If the reports were not scored on the same split, labels, and test
            rows.
    """
    _check_comparable(headline, ablated)
    drop = headline.macro_f1 - ablated.macro_f1
    change = f"({headline.macro_f1:.3f} to {ablated.macro_f1:.3f})"
    if drop > material_drop:
        interpretation = (
            f"Removing {removed} lowers macro-F1 by {drop:.3f} {change}, above the "
            f"{material_drop} materiality threshold: the headline depends on information "
            "unique to those features. They carry both muscle and neural activity, which "
            f"scalp EEG cannot separate, so {drop:.3f} bounds their combined contribution "
            "from above and does not say how much of it is EMG."
        )
    elif drop < -material_drop:
        interpretation = (
            f"Removing {removed} raises macro-F1 by {-drop:.3f} {change}: the headline "
            "does not depend on information unique to those features."
        )
    else:
        interpretation = (
            f"Removing {removed} changes macro-F1 by {-drop:+.3f} {change}, within the "
            f"{material_drop} materiality threshold: information unique to those features, "
            "muscle and neural together, is not a material part of the headline."
        )
    return {
        "removed": removed,
        "headline_macro_f1": headline.macro_f1,
        "ablated_macro_f1": ablated.macro_f1,
        "drop_headline_minus_ablated": drop,
        "material": drop > material_drop,
        "material_drop": material_drop,
        "interpretation": interpretation,
        "scope": ABLATION_SCOPE,
    }


def _write_confusion_png(report: IdentificationReport, path: Path) -> None:
    n_classes = len(report.labels)
    ramp = LinearSegmentedColormap.from_list("sequential_blue", _SEQUENTIAL_BLUE)
    colormap = ramp.with_extremes(bad=_SURFACE)
    counts = np.ma.masked_equal(report.confusion, 0)

    step = max(1, math.ceil(n_classes / 10))
    positions = np.arange(0, n_classes, step)
    tick_labels = [str(report.labels[i]) for i in positions]
    shuffled = (
        "not run"
        if report.shuffled_label_macro_f1 is None
        else f"{report.shuffled_label_macro_f1:.3f}"
    )

    with rc_context({"font.family": _FONT}):
        figure = Figure(figsize=(7.6, 7.2), dpi=150, facecolor=_SURFACE)
        FigureCanvasAgg(figure)
        axes = figure.add_subplot(facecolor=_SURFACE)
        image = axes.imshow(
            counts,
            cmap=colormap,
            vmin=1,
            vmax=max(1, int(report.confusion.max())),
            interpolation="nearest",
        )
        axes.set_xticks(positions, tick_labels)
        axes.set_yticks(positions, tick_labels)
        axes.tick_params(colors=_INK_MUTED, labelsize=8, length=0)
        for spine in axes.spines.values():
            spine.set_visible(False)
        axes.set_xlabel("Predicted subject", color=_INK_SECONDARY, fontsize=9)
        axes.set_ylabel("True subject", color=_INK_SECONDARY, fontsize=9)

        colorbar = figure.colorbar(image, ax=axes, fraction=0.046, pad=0.03)
        colorbar.outline.set_visible(False)
        colorbar.ax.tick_params(colors=_INK_MUTED, labelsize=8, length=0)
        colorbar.set_label("Test windows (blank = 0)", color=_INK_SECONDARY, fontsize=9)

        figure.subplots_adjust(left=0.09, right=0.95, bottom=0.07, top=0.86)
        figure.text(
            0.09,
            0.965,
            f"Confusion matrix: {report.split_kind.replace('_', '-')} split, "
            f"{report.normalization.replace('_', ' ')} band power",
            color=_INK_PRIMARY,
            fontsize=11,
            fontweight="bold",
            ha="left",
            va="top",
        )
        figure.text(
            0.09,
            0.925,
            f"macro-F1 {report.macro_f1:.3f}    chance {report.chance_level:.3f}    "
            f"shuffled-label control {shuffled}    {n_classes} subjects, "
            f"{report.n_test} test windows",
            color=_INK_SECONDARY,
            fontsize=9,
            ha="left",
            va="top",
        )
        figure.savefig(path, facecolor=_SURFACE)


def write_report(
    report: IdentificationReport,
    out_dir: Path,
    *,
    prefix: str,
) -> None:
    """Write the committed Phase 1 artifacts.

    Writes {prefix}_metrics.json, {prefix}_per_class_f1.csv,
    {prefix}_confusion_matrix.csv, and {prefix}_confusion_matrix.png. These are the
    Phase 1 exit criteria and are committed to the repo. The CSV is the table view of
    the PNG: an 89-by-89 heatmap shows structure, the CSV holds the counts, and it
    diffs cleanly in git.

    Args:
        report: The report to serialize.
        out_dir: The artifacts/ directory. Created if absent.
        prefix: Distinguishes runs, e.g. "relative_cross_condition".
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "split_kind": report.split_kind,
        "normalization": report.normalization,
        "macro_f1": report.macro_f1,
        "chance_level": report.chance_level,
        "shuffled_label_macro_f1": report.shuffled_label_macro_f1,
        "n_classes": len(report.labels),
        "n_train": report.n_train,
        "n_test": report.n_test,
        "n_test_not_ok": report.n_test_not_ok,
        "n_subjects_enrollable": report.n_subjects_enrollable,
        "n_subjects_impostor_holdout": report.n_subjects_impostor_holdout,
        "config_fingerprint": report.config_fingerprint,
    }
    (out_dir / f"{prefix}_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )

    with (out_dir / f"{prefix}_per_class_f1.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["subject_id", "f1"])
        for label, f1 in zip(report.labels, report.per_class_f1, strict=True):
            writer.writerow([label, f"{f1:.6f}"])

    with (out_dir / f"{prefix}_confusion_matrix.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["true\\predicted", *report.labels])
        for label, row in zip(report.labels, report.confusion, strict=True):
            writer.writerow([label, *row.tolist()])

    _write_confusion_png(report, out_dir / f"{prefix}_confusion_matrix.png")


@dataclass(frozen=True)
class RecordingQuality:
    """Quality-mask flag counts for one recording.

    Reported, not used to exclude windows from scoring (D-015). The frontal count
    backs the EOG confound check (D-004b).

    Attributes:
        subject_id: PhysioNet subject number.
        run: PhysioNet run number.
        condition: "eyes_open" or "eyes_closed".
        n_windows: Windows in the recording.
        n_windows_not_ok: Windows over the bad-channel fraction.
        n_windows_any_flag: Windows with at least one flagged channel.
        n_windows_frontal_flag: Windows with at least one frontal channel flagged.
    """

    subject_id: int
    run: int
    condition: str
    n_windows: int
    n_windows_not_ok: int
    n_windows_any_flag: int
    n_windows_frontal_flag: int


def summarize_quality(
    matrices: Sequence[FeatureMatrix],
    *,
    frontal_channels: Sequence[str] = FRONTAL_EOG_CHANNELS,
) -> list[RecordingQuality]:
    """Count quality flags per recording.

    Channel order is read from feature_names (the "mode:channel:band" contract),
    which is the order of QualityReport.channel_ok columns.

    Args:
        matrices: One per recording, carrying subject_id, run, and condition.
        frontal_channels: Channels counted as frontal. A name that is not present is
            an error, not a zero -- a spelling mismatch would otherwise report a
            clean frontal record.

    Returns:
        One RecordingQuality per matrix, in input order.

    Raises:
        ValueError: If a matrix lacks provenance, its channel count disagrees with
            its quality mask, or a frontal channel is absent from its features.
    """
    rows: list[RecordingQuality] = []
    for matrix in matrices:
        if matrix.subject_id is None or matrix.run is None or matrix.condition is None:
            raise ValueError("quality summaries need subject_id, run, and condition")
        channels = feature_channels(matrix.feature_names)
        bad = ~matrix.quality.channel_ok
        if len(channels) != bad.shape[1]:
            raise ValueError(
                f"subject {matrix.subject_id} run {matrix.run}: {len(channels)} channels in "
                f"feature_names but {bad.shape[1]} in the quality mask"
            )
        missing = [channel for channel in frontal_channels if channel not in channels]
        if missing:
            raise ValueError(f"frontal channels not present in the features: {missing}")
        frontal = [channels.index(channel) for channel in frontal_channels]

        rows.append(
            RecordingQuality(
                subject_id=matrix.subject_id,
                run=matrix.run,
                condition=matrix.condition,
                n_windows=int(matrix.quality.window_ok.size),
                n_windows_not_ok=int((~matrix.quality.window_ok).sum()),
                n_windows_any_flag=int(bad.any(axis=1).sum()),
                n_windows_frontal_flag=int(bad[:, frontal].any(axis=1).sum()),
            )
        )
    return rows


def write_quality_report(
    rows: Sequence[RecordingQuality],
    out_dir: Path,
    *,
    filename: str = QUALITY_REPORT_FILENAME,
) -> None:
    """Write per-recording quality-flag counts and rates as CSV.

    One row per subject per condition, so flag rates can be compared across
    subjects and between eyes-open and eyes-closed.

    Args:
        rows: Output of summarize_quality.
        out_dir: The artifacts/ directory. Created if absent.
        filename: Output file name.
    """

    def fraction(count: int, total: int) -> str:
        return f"{count / total:.4f}" if total else "0.0000"

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / filename).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "subject_id",
                "run",
                "condition",
                "n_windows",
                "n_windows_not_ok",
                "fraction_not_ok",
                "n_windows_any_flag",
                "fraction_any_flag",
                "n_windows_frontal_flag",
                "fraction_frontal_flag",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.subject_id,
                    row.run,
                    row.condition,
                    row.n_windows,
                    row.n_windows_not_ok,
                    fraction(row.n_windows_not_ok, row.n_windows),
                    row.n_windows_any_flag,
                    fraction(row.n_windows_any_flag, row.n_windows),
                    row.n_windows_frontal_flag,
                    fraction(row.n_windows_frontal_flag, row.n_windows),
                ]
            )
