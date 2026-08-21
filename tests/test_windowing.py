"""Window index arithmetic -- the part that is easy to get wrong by one."""

import pytest

pytestmark = pytest.mark.skip(reason="TODO(phase-1): implement after contract sign-off")


def test_window_count_at_fifty_percent_overlap() -> None:
    """60 s at 160 Hz, 2 s windows, 50% overlap yields 59 windows."""


def test_no_window_runs_past_the_end() -> None:
    """The last window start plus window length never exceeds n_samples."""


def test_partial_window_is_dropped_not_padded() -> None:
    """A trailing remainder shorter than a window is discarded, never zero-padded."""


def test_signal_shorter_than_one_window_yields_none() -> None:
    """Returns an empty index array rather than raising."""


def test_frames_are_owned_not_views() -> None:
    """Mutating one window leaves its overlapping neighbour untouched."""


def test_onsets_match_frame_contents() -> None:
    """onsets_s[i] * sfreq indexes the first sample of window i in the source."""


def test_stream_chunk_has_no_identity() -> None:
    """window_stream_chunk leaves subject_id, run, and condition as None."""


def test_stream_chunk_onsets_are_monotonic_across_chunks() -> None:
    """onset_offset_s keeps successive chunks continuous in time."""
