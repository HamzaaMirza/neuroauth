"""Composition: EDF on disk -> FeatureMatrix, through the same functions as every path."""

import dataclasses
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pytest

from neuroauth.config import FRONTAL_EOG_CHANNELS, TEMPORAL_EMG_CHANNELS, PipelineConfig
from neuroauth.dsp import pipeline
from neuroauth.dsp.features import extract_features, feature_channels
from neuroauth.dsp.io import run_condition
from neuroauth.dsp.preprocess import preprocess
from neuroauth.dsp.types import Recording
from neuroauth.dsp.windowing import window_recording
from tests.synthetic import SFREQ

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
_SUBJECT_1 = DATA_DIR / "MNE-eegbci-data" / "files" / "eegmmidb" / "1.0.0" / "S001"

needs_data = pytest.mark.skipif(
    not (_SUBJECT_1 / "S001R01.edf").exists(),
    reason="eegmmidb subject 1 not downloaded into data/",
)


def test_process_recording_is_exactly_the_composed_chain(synthetic_recording: Recording) -> None:
    """No hidden step: identical to calling preprocess, window, and features directly."""
    config = PipelineConfig()
    clean = preprocess(synthetic_recording.data, SFREQ, config.preprocess)
    windows = window_recording(dataclasses.replace(synthetic_recording, data=clean), config.window)
    expected = extract_features(windows, config.features, config.quality)

    actual = pipeline.process_recording(synthetic_recording, config)
    np.testing.assert_array_equal(actual.values, expected.values)
    assert actual.feature_names == expected.feature_names


def test_process_recording_shape_and_provenance(synthetic_recording: Recording) -> None:
    matrix = pipeline.process_recording(synthetic_recording, PipelineConfig())
    assert matrix.values.shape == (59, 64 * 5)
    assert (matrix.subject_id, matrix.run, matrix.condition) == (1, 1, "eyes_open")
    assert np.isfinite(matrix.values).all()


def test_process_recording_does_not_mutate_the_recording(synthetic_recording: Recording) -> None:
    before = synthetic_recording.data.copy()
    pipeline.process_recording(synthetic_recording, PipelineConfig())
    np.testing.assert_array_equal(synthetic_recording.data, before)


def test_iter_features_yields_one_matrix_per_loaded_recording(
    monkeypatch: pytest.MonkeyPatch, synthetic_recording: Recording
) -> None:
    calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def fake_loader(
        subjects: Sequence[int], data_dir: Path, *, runs: Sequence[int]
    ) -> Iterator[Recording]:
        calls.append((tuple(subjects), tuple(runs)))
        for subject in subjects:
            for run in runs:
                yield dataclasses.replace(
                    synthetic_recording, subject_id=subject, run=run, condition=run_condition(run)
                )

    monkeypatch.setattr(pipeline, "load_baseline_recordings", fake_loader)
    matrices = list(pipeline.iter_features([4, 5], Path("unused"), PipelineConfig()))
    assert [(m.subject_id, m.run) for m in matrices] == [(4, 1), (4, 2), (5, 1), (5, 2)]
    assert calls == [((4, 5), (1, 2))]


@pytest.mark.slow
@needs_data
def test_real_recording_carries_the_confound_check_channels() -> None:
    """Channel constants must match the loader's spelling, or the checks see nothing.

    Covers the frontal EOG channels (D-004b) and the temporal EMG channels (D-018).
    """
    (matrix,) = pipeline.iter_features([1], DATA_DIR, PipelineConfig(), runs=(1,))
    channels = set(feature_channels(matrix.feature_names))
    assert set(FRONTAL_EOG_CHANNELS) <= channels
    assert set(TEMPORAL_EMG_CHANNELS) <= channels
    assert matrix.values.shape == (60, 64 * 5)
    assert np.isfinite(matrix.values).all()
