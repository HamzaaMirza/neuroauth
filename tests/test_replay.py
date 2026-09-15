"""Replay framing, the crossfaded splice used for impostor injection, and replayed sessions."""

import dataclasses

import numpy as np
import pytest

from neuroauth.session import replay
from neuroauth.session.replay import replay_chunks, replay_session, splice
from tests.session_support import (
    CHANNELS,
    THRESHOLDS,
    SessionParts,
    build_session_parts,
    install_logic_double,
    recording,
)
from tests.synthetic import SFREQ


@pytest.fixture(scope="module")
def parts() -> SessionParts:
    return build_session_parts()


def test_replay_constants_are_pinned() -> None:
    """Fixed before any Phase 2 result (D-016 rule)."""
    assert (replay.SWAP_AT_S, replay.CROSSFADE_S) == (30.0, 0.5)
    assert (replay.DETECTION_HORIZON_S, replay.REPLAY_CHUNK_S) == (25.0, 0.1)


def test_replay_chunks_cover_the_stream_in_order() -> None:
    data = np.arange(2 * 35, dtype=np.float64).reshape(2, 35)
    chunks = list(replay_chunks(data, 160.0, chunk_s=0.1))
    assert [c.shape[1] for c in chunks] == [16, 16, 3]
    np.testing.assert_array_equal(np.concatenate(chunks, axis=1), data)
    with pytest.raises(ValueError, match="below one sample"):
        list(replay_chunks(data, 160.0, chunk_s=0.001))


def test_splice_keeps_both_sources_intact_outside_the_crossfade() -> None:
    first, second = recording(5, 2, seed=1), recording(6, 2, seed=2)
    spliced = splice(first, second, swap_at_s=10.0, crossfade_s=0.5)
    swap, fade = round(10.0 * SFREQ), round(0.5 * SFREQ)
    np.testing.assert_array_equal(spliced[:, :swap], first.data[:, :swap])
    np.testing.assert_array_equal(spliced[:, swap + fade :], second.data[:, swap + fade :])
    assert spliced.shape == second.data.shape
    middle = swap + fade // 2
    expected = 0.5 * (first.data[:, middle] + second.data[:, middle])
    np.testing.assert_allclose(spliced[:, middle], expected, rtol=0.05, atol=1e-7)


def test_self_splice_is_allowed_only_at_a_different_offset() -> None:
    probe = recording(5, 2, seed=1)
    with pytest.raises(ValueError, match="changes nothing"):
        splice(probe, probe, swap_at_s=10.0)
    spliced = splice(probe, probe, swap_at_s=10.0, second_offset_s=20.0)
    np.testing.assert_array_equal(
        spliced[:, round(10.5 * SFREQ)], probe.data[:, round(20.5 * SFREQ)]
    )


def test_splice_refuses_mismatched_recordings_and_bad_offsets() -> None:
    first = recording(5, 2, seed=1)
    renamed = dataclasses.replace(recording(6, 2, seed=2), ch_names=tuple(reversed(CHANNELS)))
    with pytest.raises(ValueError, match="channel order"):
        splice(first, renamed)
    with pytest.raises(ValueError, match="outside"):
        splice(first, recording(6, 2, seed=2), swap_at_s=39.9)


def test_replayed_session_produces_a_trace(
    parts: SessionParts, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_logic_double(monkeypatch)
    data = splice(parts.probe, parts.impostor, swap_at_s=20.0)
    trace = replay_session(
        data,
        verifier=parts.verifier,
        thresholds=THRESHOLDS,
        claimed_subject=5,
        ch_names=CHANNELS,
        sfreq=SFREQ,
        swap_at_s=20.0,
        impostor_subject=6,
    )
    assert trace.swap_at_s == 20.0 and trace.impostor_subject == 6
    assert trace.decision_times_s[0] == 6.0
    assert np.all(np.diff(trace.decision_times_s) > 0)
    assert set(trace.states) == {"active"}


def test_replay_stops_at_the_first_terminal_state(
    parts: SessionParts, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_logic_double(monkeypatch, revoke_after=4)
    trace = replay_session(
        parts.probe.data,
        verifier=parts.verifier,
        thresholds=THRESHOLDS,
        claimed_subject=5,
        ch_names=CHANNELS,
        sfreq=SFREQ,
        swap_at_s=None,
        impostor_subject=None,
    )
    assert trace.states[-1] == "revoked"
    assert len(trace.states) == 4
