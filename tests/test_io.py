"""Loader tests.

Tests that read EDF files are marked slow and skip when the eegmmidb download is
absent, so they never trigger a network fetch. Argument validation and the batch
loader's skip logic need no data and always run.
"""

from pathlib import Path

import mne
import numpy as np
import pytest

from neuroauth.dsp import io as dsp_io
from neuroauth.dsp.io import (
    RecordingLoadError,
    load_baseline_recordings,
    load_recording,
    run_condition,
)
from neuroauth.dsp.types import Recording

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
_SUBJECT_1 = DATA_DIR / "MNE-eegbci-data" / "files" / "eegmmidb" / "1.0.0" / "S001"

needs_data = pytest.mark.skipif(
    not (_SUBJECT_1 / "S001R01.edf").exists() or not (_SUBJECT_1 / "S001R02.edf").exists(),
    reason="eegmmidb subject 1 not downloaded into data/",
)


@pytest.mark.slow
@needs_data
def test_loader_strips_trailing_dots_from_channel_names() -> None:
    """Channel names come back standardized: no trailing dots, 10-10 spelling.

    eegbci.standardize runs inside the loader, so no caller ever has to remember to
    call it. This test is what stops that from regressing.
    """
    recording = load_recording(1, 1, DATA_DIR)
    assert not any(name.endswith(".") for name in recording.ch_names)
    assert "FC5" in recording.ch_names
    montage = mne.channels.make_standard_montage("standard_1005")
    assert set(recording.ch_names) <= set(montage.ch_names)


@pytest.mark.slow
@needs_data
def test_loader_returns_volts_as_float64() -> None:
    recording = load_recording(1, 2, DATA_DIR)
    assert recording.data.dtype == np.float64
    assert recording.data.shape[0] == 64
    assert recording.sfreq == 160.0
    assert (recording.subject_id, recording.run, recording.condition) == (1, 2, "eyes_closed")
    # Scalp EEG is tens of microvolts; a median in this range means volts, not uV.
    assert 1e-6 < float(np.median(np.abs(recording.data))) < 1e-3


@pytest.mark.slow
@needs_data
def test_loader_rejects_unexpected_sample_rate() -> None:
    with pytest.raises(RecordingLoadError, match="Hz"):
        load_recording(1, 1, DATA_DIR, expected_sfreq=128.0)


@pytest.mark.slow
@needs_data
def test_loader_rejects_unexpected_channel_count() -> None:
    with pytest.raises(RecordingLoadError, match="channels"):
        load_recording(1, 1, DATA_DIR, expected_n_channels=32)


@pytest.mark.parametrize(
    ("run", "expected"),
    [(1, "eyes_open"), (2, "eyes_closed"), (3, "task"), (14, "task")],
)
def test_run_condition_mapping(run: int, expected: str) -> None:
    """Run 1 is eyes_open, run 2 is eyes_closed, runs 3-14 are task."""
    assert run_condition(run) == expected


@pytest.mark.parametrize("run", [0, 15, -1])
def test_run_condition_rejects_out_of_range(run: int) -> None:
    with pytest.raises(ValueError):
        run_condition(run)


@pytest.mark.parametrize("subject", [0, 110])
def test_load_recording_rejects_out_of_range_subject(subject: int, tmp_path: Path) -> None:
    """Raised before any download is attempted."""
    with pytest.raises(ValueError):
        load_recording(subject, 1, tmp_path)


def _fake_loader(failing_subject: int):  # type: ignore[no-untyped-def]
    def fake_load_recording(subject: int, run: int, data_dir: Path) -> Recording:
        if subject == failing_subject:
            raise RecordingLoadError(f"subject {subject} is broken")
        return Recording(np.zeros((1, 10)), 160.0, ("Cz",), subject, run, run_condition(run))

    return fake_load_recording


def test_batch_loader_skips_failures_when_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bad subject is skipped, not fatal, when skip_failures is True."""
    monkeypatch.setattr(dsp_io, "load_recording", _fake_loader(failing_subject=2))
    loaded = [(r.subject_id, r.run) for r in load_baseline_recordings([1, 2, 3], Path("unused"))]
    assert loaded == [(1, 1), (1, 2), (3, 1), (3, 2)]


def test_batch_loader_raises_when_not_skipping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dsp_io, "load_recording", _fake_loader(failing_subject=2))
    with pytest.raises(RecordingLoadError):
        list(load_baseline_recordings([1, 2, 3], Path("unused"), skip_failures=False))
