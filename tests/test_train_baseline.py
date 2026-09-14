"""The baseline driver: preconditions, gating, and the artifacts it writes.

Runs on synthetic feature matrices -- no EEG is loaded.
"""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scripts import train_baseline as driver

from neuroauth.config import BANDS, Normalization
from neuroauth.dsp.types import FeatureMatrix
from neuroauth.models.baseline import BaselineConfig
from tests.synthetic import feature_matrix

CHANNELS = ("Fp1", "Fp2", "AF7", "AF8", "C3", "C4")
SUBJECTS = (1, 2, 4, 5, 6, 7)
HOLDOUT = frozenset({3, 11})
FAST = BaselineConfig(n_estimators=40, n_jobs=1)
N_WINDOWS = 30


def _cohort(
    normalization: Normalization, subjects: tuple[int, ...] = SUBJECTS
) -> list[FeatureMatrix]:
    """Two recordings per subject around a subject-specific center."""
    prefix = "rel" if normalization == "relative" else "abs"
    names = tuple(f"{prefix}:{channel}:{band}" for channel in CHANNELS for band in BANDS)
    rng = np.random.default_rng(0)
    centers = {subject: rng.normal(0.0, 3.0, size=len(names)) for subject in subjects}
    return [
        feature_matrix(
            subject_id=subject,
            run=run,
            condition=condition,
            n_windows=N_WINDOWS,
            feature_names=names,
            values=centers[subject] + rng.normal(size=(N_WINDOWS, len(names))),
        )
        for subject in subjects
        for run, condition in ((1, "eyes_open"), (2, "eyes_closed"))
    ]


def _run(
    tmp_path: Path, matrices: dict[Normalization, list[FeatureMatrix]] | None = None
) -> dict[str, Any]:
    return driver.run(
        matrices if matrices is not None else {n: _cohort(n) for n in driver.NORMALIZATIONS},
        holdout=HOLDOUT,
        out_dir=tmp_path / "artifacts",
        fingerprints={"relative": "fp-relative", "absolute_log": "fp-absolute"},
        window_s=2.0,
        baseline=FAST,
        provenance={"git_head": "test-head"},
    )


def test_run_writes_every_artifact(tmp_path: Path) -> None:
    summary = _run(tmp_path)
    expected = {
        f"{normalization}_{split_kind}_{suffix}"
        for normalization in driver.NORMALIZATIONS
        for split_kind in driver.SPLIT_KINDS
        for suffix in (
            "metrics.json",
            "per_class_f1.csv",
            "confusion_matrix.csv",
            "confusion_matrix.png",
        )
    } | {
        "normalization_comparison.json",
        "importance.json",
        "quality_flag_rates.csv",
        "leakage_demonstration.json",
        "run_summary.json",
    }
    assert {path.name for path in (tmp_path / "artifacts").iterdir()} == expected
    assert (summary["headline"]["normalization"], summary["headline"]["split_kind"]) == (
        "relative",
        "cross_condition",
    )
    assert summary["git_head"] == "test-head"
    assert summary["a_priori_thresholds"] == {
        "control_max_chance_ratio": 3.0,
        "material_delta": 0.05,
    }


def test_no_artifact_mentions_accuracy(tmp_path: Path) -> None:
    _run(tmp_path)
    for path in (tmp_path / "artifacts").iterdir():
        if path.suffix in {".json", ".csv"}:
            assert "accuracy" not in path.read_text(encoding="utf-8").lower(), path.name


def test_leakage_demonstration_is_labelled_and_kept_apart(tmp_path: Path) -> None:
    summary = _run(tmp_path)
    out = tmp_path / "artifacts"
    leakage = json.loads((out / driver.LEAKAGE_ARTIFACT).read_text(encoding="utf-8"))
    assert "DELIBERATELY LEAKY" in leakage["label"]
    assert leakage["leaky_test_windows_sharing_samples_with_train"] > 0
    assert not any(driver.LEAKY_SPLIT_KIND in path.name for path in out.iterdir())
    assert all(entry["split_kind"] in driver.SPLIT_KINDS for entry in summary["runs"])


def test_importance_reports_frontal_share_against_uniform(tmp_path: Path) -> None:
    _run(tmp_path)
    importance = json.loads(
        (tmp_path / "artifacts" / "importance.json").read_text(encoding="utf-8")
    )
    entry = importance["relative_cross_condition"]
    assert entry["trained_on"] == "eyes_open"
    assert entry["frontal_uniform_share"] == pytest.approx(4 / len(CHANNELS))
    assert 0.0 <= entry["frontal_importance_share"] <= 1.0


def test_failed_control_aborts_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(driver, "run_shuffled_label_control", lambda *args, **kwargs: 0.99)
    with pytest.raises(AssertionError, match="shuffled-label"):
        _run(tmp_path)
    assert not (tmp_path / "artifacts").exists()


def test_holdout_subject_aborts_before_writing(tmp_path: Path) -> None:
    leaked = {n: _cohort(n, subjects=(*SUBJECTS[:-1], 3)) for n in driver.NORMALIZATIONS}
    with pytest.raises(AssertionError, match=r"\[3\]"):
        _run(tmp_path, leaked)
    assert not (tmp_path / "artifacts").exists()


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        cwd=repo,
        check=True,
        capture_output=True,
    )


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_holdout_precondition_needs_a_committed_unmodified_file(tmp_path: Path) -> None:
    """Throwaway repository in tmp_path; the project repository is never touched."""
    _git(tmp_path, "init", "-q")
    holdout = tmp_path / "config" / "impostor_holdout.json"
    holdout.parent.mkdir()
    holdout.write_text("{}\n", encoding="utf-8")

    with pytest.raises(driver.PreconditionError, match="not committed"):
        driver.check_holdout_committed(holdout, tmp_path)

    _git(tmp_path, "add", "config/impostor_holdout.json")
    with pytest.raises(driver.PreconditionError, match="not committed"):
        driver.check_holdout_committed(holdout, tmp_path)

    _git(tmp_path, "commit", "-q", "-m", "holdout")
    assert len(driver.check_holdout_committed(holdout, tmp_path)) == 40

    holdout.write_text('{"edited": true}\n', encoding="utf-8")
    with pytest.raises(driver.PreconditionError, match="differs"):
        driver.check_holdout_committed(holdout, tmp_path)


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_working_tree_changes_report_code_but_not_artifacts(tmp_path: Path) -> None:
    """A dirty tree is reported; regenerated artifacts on their own are not."""
    _git(tmp_path, "init", "-q")
    (tmp_path / "code.py").write_text("x = 1\n", encoding="utf-8")
    _git(tmp_path, "add", "code.py")
    _git(tmp_path, "commit", "-q", "-m", "code")
    assert driver.working_tree_changes(tmp_path) == []

    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts" / "run_summary.json").write_text("{}\n", encoding="utf-8")
    assert driver.working_tree_changes(tmp_path) == []

    (tmp_path / "code.py").write_text("x = 2\n", encoding="utf-8")
    (tmp_path / "new_module.py").write_text("y = 1\n", encoding="utf-8")
    assert sorted(driver.working_tree_changes(tmp_path)) == ["code.py", "new_module.py"]
