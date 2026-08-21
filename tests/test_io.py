"""Loader tests. These need the eegmmidb download in data/ and are marked slow."""

import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skip(reason="TODO(phase-1): implement after contract sign-off"),
]


def test_loader_strips_trailing_dots_from_channel_names() -> None:
    """Recording.ch_names contains Fc5, never Fc5 with a trailing dot.

    eegbci.standardize runs inside the loader, so no caller ever has to remember to
    call it. This test is what stops that from regressing.
    """


def test_loader_rejects_unexpected_sample_rate() -> None:
    """A header disagreeing with expected_sfreq raises RecordingLoadError."""


def test_loader_rejects_unexpected_channel_count() -> None:
    """A header disagreeing with expected_n_channels raises RecordingLoadError."""


def test_run_condition_mapping() -> None:
    """Run 1 is eyes_open, run 2 is eyes_closed, runs 3-14 are task."""


def test_batch_loader_skips_failures_when_asked() -> None:
    """A bad subject is skipped, not fatal, when skip_failures is True."""
