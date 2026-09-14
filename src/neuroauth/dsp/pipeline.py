"""End-to-end composition: EDF on disk -> FeatureMatrix in memory.

Composes the public dsp functions and adds nothing of its own, so a feature vector
produced here is identical to one produced by calling those functions directly.
"""

import dataclasses
from collections.abc import Iterator, Sequence
from pathlib import Path

from neuroauth.config import PipelineConfig
from neuroauth.dsp.features import extract_features
from neuroauth.dsp.io import BASELINE_RUNS, load_baseline_recordings
from neuroauth.dsp.preprocess import preprocess
from neuroauth.dsp.types import FeatureMatrix, Recording
from neuroauth.dsp.windowing import window_recording


def process_recording(recording: Recording, config: PipelineConfig) -> FeatureMatrix:
    """Preprocess, window, and extract features for one recording.

    Args:
        recording: As returned by load_recording. Not mutated.
        config: The full pipeline config.

    Returns:
        A FeatureMatrix carrying the provenance of the recording.
    """
    clean = preprocess(recording.data, recording.sfreq, config.preprocess)
    windows = window_recording(dataclasses.replace(recording, data=clean), config.window)
    return extract_features(windows, config.features, config.quality)


def iter_features(
    subjects: Sequence[int],
    data_dir: Path,
    config: PipelineConfig,
    *,
    runs: Sequence[int] = BASELINE_RUNS,
) -> Iterator[FeatureMatrix]:
    """Stream FeatureMatrices, one recording at a time.

    Raw arrays are released after each recording, so peak memory stays at the
    windows of one recording (~10 MB) rather than the whole corpus (~2 GB). The
    resulting features for all baseline recordings total roughly 33 MB.

    Args:
        subjects: Subject numbers. The caller is responsible for having already
            excluded the impostor holdout -- see neuroauth.cohorts.
        data_dir: MNE cache root.
        config: The full pipeline config.
        runs: Which runs to include.

    Yields:
        One FeatureMatrix per successfully loaded recording.
    """
    for recording in load_baseline_recordings(subjects, data_dir, runs=runs):
        yield process_recording(recording, config)
