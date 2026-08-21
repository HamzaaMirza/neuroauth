"""EDF loading for PhysioNet eegmmidb.

The only module in the package that imports MNE. Everything downstream takes numpy
arrays, so the same preprocessing code runs on a live WebSocket stream in Phase 2
with no MNE object in sight.

Data: PhysioNet EEG Motor Movement/Imagery Database v1.0.0.
License: Open Data Commons Attribution (ODC-By) v1.0.
"""

from collections.abc import Iterator, Sequence
from pathlib import Path

from neuroauth.dsp.types import Condition, Recording

BASELINE_RUNS: tuple[int, ...] = (1, 2)
"""R01 eyes-open, R02 eyes-closed. The resting-state runs -- the realistic auth
input. Task runs R03-R14 are a Phase 2 robustness holdout."""


class RecordingLoadError(Exception):
    """Raised when an EDF file cannot be read or fails structural validation.

    Loading is not the inference path, so raising here is correct. The batch loader
    below catches this and skips rather than propagating.
    """


def run_condition(run: int) -> Condition:
    """Map a PhysioNet run number to its recording condition.

    Args:
        run: 1, 2, or 3-14.

    Returns:
        "eyes_open" for run 1, "eyes_closed" for run 2, "task" for 3-14.

    Raises:
        ValueError: If run is outside 1-14.
    """
    raise NotImplementedError("TODO(phase-1): run -> condition mapping")


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

    Applies eegbci.standardize(raw) internally. The eegmmidb EDF headers carry
    trailing dots ("Fc5.", "C5..", "Fcz.") and the 10-10 montage will not match
    without it. Callers must never standardize: by the time a Recording exists, its
    ch_names are already clean.

    Args:
        subject: PhysioNet subject number, 1-109.
        run: PhysioNet run number, 1-14.
        data_dir: Root of the MNE dataset cache.
        expected_sfreq: Asserted against the file header.
        expected_n_channels: Asserted against the file header.

    Returns:
        A Recording with data in volts, shape (n_channels, n_samples).

    Raises:
        RecordingLoadError: Download failed, EDF unreadable, or the header disagrees
            with expected_sfreq / expected_n_channels.
    """
    raise NotImplementedError("TODO(phase-1): EDF load + standardize + validate")


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
        subjects: Subject numbers to load.
        data_dir: Root of the MNE dataset cache.
        runs: Which runs. Defaults to the two baseline runs.
        skip_failures: Log and continue past a RecordingLoadError rather than
            aborting the batch. Subjects dropped this way are recorded in
            docs/DECISIONS.md.

    Yields:
        Recordings in (subject, run) order.
    """
    raise NotImplementedError("TODO(phase-1): streaming batch loader")
