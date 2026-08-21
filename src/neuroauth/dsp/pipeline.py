"""End-to-end composition: EDF on disk -> FeatureMatrix in memory."""

from collections.abc import Iterator, Sequence
from pathlib import Path

from neuroauth.config import PipelineConfig
from neuroauth.dsp.types import FeatureMatrix, Recording


def process_recording(recording: Recording, config: PipelineConfig) -> FeatureMatrix:
    """Preprocess, window, and extract features for one recording.

    Args:
        recording: As returned by load_recording. Not mutated.
        config: The full pipeline config.

    Returns:
        A FeatureMatrix carrying the provenance of the recording.
    """
    raise NotImplementedError("TODO(phase-1): compose the per-recording pipeline")


def iter_features(
    subjects: Sequence[int],
    data_dir: Path,
    config: PipelineConfig,
    *,
    runs: Sequence[int] = (1, 2),
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
    raise NotImplementedError("TODO(phase-1): streaming feature extraction")
