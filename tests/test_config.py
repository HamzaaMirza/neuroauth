"""PipelineConfig.fingerprint: stable, sensitive to every setting, and order-aware."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from neuroauth.config import (
    BANDS,
    ContextConfig,
    FeatureConfig,
    PipelineConfig,
    PreprocessConfig,
    QualityConfig,
    StreamingConfig,
    WindowConfig,
)


def test_fingerprint_is_deterministic_hex() -> None:
    fingerprint = PipelineConfig().fingerprint()
    assert fingerprint == PipelineConfig().fingerprint()
    assert len(fingerprint) == 16
    int(fingerprint, 16)


def test_equal_values_give_equal_fingerprints() -> None:
    """500e-6 and 5e-4 are the same float, so the configs are the same config."""
    spelled_differently = PipelineConfig(quality=QualityConfig(max_peak_to_peak_v=5e-4))
    assert spelled_differently.fingerprint() == PipelineConfig().fingerprint()


@pytest.mark.parametrize(
    "changed",
    [
        PipelineConfig(features=FeatureConfig(normalization="absolute_log")),
        PipelineConfig(window=WindowConfig(window_s=1.0)),
        PipelineConfig(quality=QualityConfig(max_peak_to_peak_v=250e-6)),
        PipelineConfig(preprocess=PreprocessConfig(common_average_reference=True)),
    ],
    ids=["normalization", "window", "quality", "car"],
)
def test_any_change_changes_the_fingerprint(changed: PipelineConfig) -> None:
    assert changed.fingerprint() != PipelineConfig().fingerprint()


def test_band_order_changes_the_fingerprint() -> None:
    """Band order fixes feature column order, so it must be part of the fingerprint."""
    reordered = FeatureConfig(bands=dict(reversed(list(BANDS.items()))))
    assert PipelineConfig(features=reordered).fingerprint() != PipelineConfig().fingerprint()


def test_phase1_fingerprints_still_reproduce() -> None:
    """Phase 2 added StreamingConfig without touching PipelineConfig (D-020)."""
    summary = json.loads(
        (Path(__file__).resolve().parents[1] / "artifacts" / "run_summary.json").read_text("utf-8")
    )
    for normalization in ("relative", "absolute_log"):
        config = PipelineConfig(features=FeatureConfig(normalization=normalization))
        assert config.fingerprint() == summary["config_fingerprints"][normalization]


def test_streaming_fingerprint_is_namespaced_and_margin_sensitive() -> None:
    assert StreamingConfig().fingerprint() == StreamingConfig().fingerprint()
    assert StreamingConfig().fingerprint() != PipelineConfig().fingerprint()
    wider = StreamingConfig(context=ContextConfig(right_margin_s=3.0))
    assert wider.fingerprint() != StreamingConfig().fingerprint()


def test_fingerprint_is_stable_across_processes() -> None:
    """No dependence on hash randomization or anything else process-local."""
    code = "from neuroauth.config import PipelineConfig; print(PipelineConfig().fingerprint())"
    outputs = {
        subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        for seed in ("0", "12345")
    }
    assert outputs == {PipelineConfig().fingerprint()}
