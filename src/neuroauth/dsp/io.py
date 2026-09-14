"""EDF loading for PhysioNet eegmmidb.

The only module in the package that imports MNE. Everything downstream takes numpy
arrays, so the same preprocessing code runs on a live WebSocket stream in Phase 2
with no MNE object in sight.

Data: PhysioNet EEG Motor Movement/Imagery Database v1.0.0.
License: Open Data Commons Attribution (ODC-By) v1.0.
"""

import logging
from collections.abc import Iterator, Sequence
from pathlib import Path

import mne
import numpy as np
from mne.datasets import eegbci

from neuroauth.dsp.types import Condition, Recording

logger = logging.getLogger(__name__)

BASELINE_RUNS: tuple[int, ...] = (1, 2)
"""R01 eyes-open, R02 eyes-closed. The resting-state runs -- the realistic auth
input. Task runs R03-R14 are a Phase 2 robustness holdout."""

N_SUBJECTS = 109
N_RUNS = 14


class RecordingLoadError(Exception):
    """Raised when an EDF file cannot be read or fails structural validation.

    Loading is not the inference path, so raising here is correct. The batch loader
    below catches this and skips rather than propagating.
    """


def _label(subject: int, run: int) -> str:
    return f"S{subject:03d}R{run:02d}"


def run_condition(run: int) -> Condition:
    """Map a PhysioNet run number to its recording condition.

    Args:
        run: 1, 2, or 3-14.

    Returns:
        "eyes_open" for run 1, "eyes_closed" for run 2, "task" for 3-14.

    Raises:
        ValueError: If run is outside 1-14.
    """
    if run == 1:
        return "eyes_open"
    if run == 2:
        return "eyes_closed"
    if 3 <= run <= N_RUNS:
        return "task"
    raise ValueError(f"run must be in 1-{N_RUNS}, got {run}")


def load_recording(
    subject: int,
    run: int,
    data_dir: Path,
    *,
    expected_sfreq: float = 160.0,
    expected_n_channels: int = 64,
) -> Recording:
    """Load and validate one EDF run for one subject.

    Downloads via mne.datasets.eegbci.load_data if not already cached in data_dir.
    A cached file is used as-is, with no network access.

    Applies eegbci.standardize(raw) internally. The eegmmidb EDF headers carry
    trailing dots and non-standard casing ("Fc5.", "Fcz.") and the 10-10 montage will
    not match without it; after standardizing they read "FC5", "FCz". Callers must
    never standardize: by the time a Recording exists, its ch_names are already
    clean.

    The header is validated before the signal is read, so a rejected file costs no
    data I/O.

    Args:
        subject: PhysioNet subject number, 1-109.
        run: PhysioNet run number, 1-14.
        data_dir: Root of the MNE dataset cache.
        expected_sfreq: Asserted against the file header.
        expected_n_channels: Asserted against the file header.

    Returns:
        A Recording with data in volts, shape (n_channels, n_samples).

    Raises:
        ValueError: If subject or run is out of range. Checked before any download,
            because a bad argument is a caller bug, not a data problem.
        RecordingLoadError: Download failed, EDF unreadable, or the header disagrees
            with expected_sfreq / expected_n_channels.
    """
    if not 1 <= subject <= N_SUBJECTS:
        raise ValueError(f"subject must be in 1-{N_SUBJECTS}, got {subject}")
    condition = run_condition(run)
    label = _label(subject, run)

    try:
        paths = eegbci.load_data(
            subjects=[subject], runs=[run], path=str(data_dir), update_path=False, verbose="ERROR"
        )
    except Exception as exc:
        raise RecordingLoadError(f"{label}: download failed: {exc}") from exc

    try:
        raw = mne.io.read_raw_edf(paths[0], preload=False, verbose="ERROR")
    except Exception as exc:
        raise RecordingLoadError(f"{label}: EDF header unreadable: {exc}") from exc

    sfreq = float(raw.info["sfreq"])
    if sfreq != expected_sfreq:
        raise RecordingLoadError(f"{label}: sampled at {sfreq} Hz, expected {expected_sfreq} Hz")
    if len(raw.ch_names) != expected_n_channels:
        raise RecordingLoadError(
            f"{label}: {len(raw.ch_names)} channels, expected {expected_n_channels}"
        )

    try:
        raw.load_data(verbose="ERROR")
    except Exception as exc:
        raise RecordingLoadError(f"{label}: EDF signal unreadable: {exc}") from exc

    eegbci.standardize(raw)

    return Recording(
        data=np.asarray(raw.get_data(), dtype=np.float64),
        sfreq=sfreq,
        ch_names=tuple(raw.ch_names),
        subject_id=subject,
        run=run,
        condition=condition,
    )


def load_baseline_recordings(
    subjects: Sequence[int],
    data_dir: Path,
    *,
    runs: Sequence[int] = BASELINE_RUNS,
    skip_failures: bool = True,
) -> Iterator[Recording]:
    """Yield baseline recordings one at a time.

    A generator, not a list: 218 recordings held simultaneously is roughly 1 GB of
    float64 before windowing doubles it. Callers extract features per recording and
    release the raw array.

    Args:
        subjects: Subject numbers to load, iterated in the order given.
        data_dir: Root of the MNE dataset cache.
        runs: Which runs. Defaults to the two baseline runs.
        skip_failures: Log and continue past a RecordingLoadError rather than
            aborting the batch. Subjects dropped this way are recorded in
            docs/DECISIONS.md. A ValueError (bad argument) always propagates.

    Yields:
        Recordings in (subject, run) order.
    """
    for subject in subjects:
        for run in runs:
            try:
                recording = load_recording(subject, run, data_dir)
            except RecordingLoadError as exc:
                if not skip_failures:
                    raise
                logger.warning("skipping recording: %s", exc)
                continue
            yield recording
