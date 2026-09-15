"""The session runtime: frames, warm-up, gaps, terminal states, and bit-for-bit agreement
between a replayed stream and offline scoring (D-020, end to end)."""

import struct
from collections.abc import Mapping

import numpy as np
import pytest

from neuroauth.dsp.streaming import process_recording_bounded
from neuroauth.session import logic
from neuroauth.session.logic import SessionTransition
from neuroauth.session.runtime import (
    FRAME_HEADER,
    FRAME_MAGIC,
    MAX_FRAME_SAMPLES,
    FrameError,
    SampleFrame,
    SessionRuntime,
    decode_frame,
    encode_frame,
    handle_frame,
    handle_stop,
    start_runtime,
)
from neuroauth.verification.scoring import score_windows
from tests.session_support import (
    STREAMING,
    THRESHOLDS,
    SessionParts,
    build_session_parts,
    install_logic_double,
    start_message,
)
from tests.synthetic import SFREQ

FRAME = round(0.1 * SFREQ)


@pytest.fixture(scope="module")
def parts() -> SessionParts:
    return build_session_parts()


def start(parts: SessionParts) -> SessionRuntime:
    runtime = start_runtime("s-1", start_message(parts), parts.verifier, THRESHOLDS)
    assert isinstance(runtime, SessionRuntime)
    return runtime


def feed(
    runtime: SessionRuntime, data: np.ndarray, *, first_seq: int = 0
) -> tuple[SessionRuntime, list[Mapping[str, object]], list[SessionTransition], bool]:
    """Send data in the largest frames the protocol allows, stopping when the session closes."""
    messages: list[Mapping[str, object]] = []
    transitions: list[SessionTransition] = []
    for index, position in enumerate(range(0, data.shape[1], MAX_FRAME_SAMPLES)):
        frame = encode_frame(first_seq + index, data[:, position : position + MAX_FRAME_SAMPLES])
        output = handle_frame(runtime, frame)
        runtime = output.runtime
        messages += output.messages
        transitions += output.transitions
        if output.close:
            return runtime, messages, transitions, True
    return runtime, messages, transitions, False


def test_the_authors_session_logic_is_still_a_stub() -> None:
    with pytest.raises(NotImplementedError):
        logic.initial_session_state()


def test_frames_round_trip_bit_for_bit() -> None:
    samples = np.random.default_rng(0).normal(size=(8, 16)) * 1e-5
    decoded = decode_frame(encode_frame(7, samples), 8)
    assert isinstance(decoded, SampleFrame)
    assert decoded.seq == 7
    np.testing.assert_array_equal(decoded.samples, samples)


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        (b"NAF", "short_header"),
        (FRAME_HEADER.pack(b"XXXX", 0, 8, 1) + bytes(64), "bad_magic"),
        (FRAME_HEADER.pack(FRAME_MAGIC, 0, 4, 1) + bytes(32), "channel_mismatch"),
        (FRAME_HEADER.pack(FRAME_MAGIC, 0, 8, MAX_FRAME_SAMPLES + 1), "too_many_samples"),
        (FRAME_HEADER.pack(FRAME_MAGIC, 0, 8, 2) + bytes(64), "length_mismatch"),
    ],
)
def test_decode_never_raises_on_malformed_frames(payload: bytes, code: str) -> None:
    decoded = decode_frame(payload, 8)
    assert isinstance(decoded, FrameError)
    assert decoded.code == code


def test_encode_refuses_frames_outside_the_protocol() -> None:
    with pytest.raises(ValueError):
        encode_frame(0, np.zeros(8))
    with pytest.raises(ValueError):
        encode_frame(0, np.zeros((8, MAX_FRAME_SAMPLES + 1)))
    with pytest.raises(ValueError):
        encode_frame(-1, np.zeros((8, 1)))


@pytest.mark.parametrize(
    ("change", "code"),
    [
        ({"type": "hello"}, "bad_start"),
        ({"claimed_subject": "eegmmidb-S006"}, "bad_start"),
        ({"sfreq": True}, "bad_start"),
        ({"sfreq": 128.0}, "channel_mismatch"),
        (
            {"ch_names": ["Ch02", "Ch01", "Ch03", "Ch04", "Ch05", "Ch06", "Ch07", "Ch08"]},
            "channel_mismatch",
        ),
    ],
)
def test_start_is_validated_against_the_model(
    parts: SessionParts, monkeypatch: pytest.MonkeyPatch, change: dict[str, object], code: str
) -> None:
    install_logic_double(monkeypatch)
    refused = start_runtime("s-1", {**start_message(parts), **change}, parts.verifier, THRESHOLDS)
    assert isinstance(refused, FrameError)
    assert refused.code == code


@pytest.mark.parametrize("frame_sizes", ["fixed", "random"])
def test_replayed_stream_scores_match_offline_scoring_bit_for_bit(
    parts: SessionParts, monkeypatch: pytest.MonkeyPatch, frame_sizes: str
) -> None:
    """Through encode_frame, decode_frame, the stream buffer, and scoring: identical to
    scoring the offline bounded-context features of the same recording."""
    install_logic_double(monkeypatch)
    data = parts.probe.data
    rng = np.random.default_rng(4)
    runtime = start(parts)
    windows: list[dict[str, object]] = []
    position, seq = 0, 0
    while position < data.shape[1]:
        size = FRAME if frame_sizes == "fixed" else int(rng.integers(1, 700))
        output = handle_frame(runtime, encode_frame(seq, data[:, position : position + size]))
        runtime = output.runtime
        windows += [dict(m) for m in output.messages if m["type"] == "window"]
        position += size
        seq += 1

    offline = score_windows(parts.verifier, process_recording_bounded(parts.probe, STREAMING))
    assert len(windows) == len(offline) > 0
    assert [w["score"] for w in windows] == [o.score for o in offline]
    assert [w["llr"] for w in windows] == [o.llr for o in offline]
    assert [w["onset_s"] for w in windows] == [o.onset_s for o in offline]
    assert windows[0]["decision_time_s"] == 6.0


def test_warm_up_status_until_the_first_window(
    parts: SessionParts, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_logic_double(monkeypatch)
    runtime = start(parts)
    output = handle_frame(runtime, encode_frame(0, parts.probe.data[:, :FRAME]))
    status = [m for m in output.messages if m["type"] == "status"]
    assert status and status[0]["warming_up"] is True
    assert status[0]["ready_in_s"] == pytest.approx(6.0 - 0.1)


def test_a_gap_resets_the_buffer_and_flags_the_next_observation(
    parts: SessionParts, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed = install_logic_double(monkeypatch)
    data = parts.probe.data
    runtime = start(parts)
    half = round(10.0 * SFREQ)
    runtime = handle_frame(runtime, encode_frame(0, data[:, :half])).runtime
    before_gap = len(observed)
    assert before_gap > 0 and not any(o.gap_before for o in observed)

    resumed = half + round(5.0 * SFREQ)
    output = handle_frame(runtime, encode_frame(5, data[:, half:resumed]))
    assert len(observed) == before_gap  # 5 s after a reset is still warming up
    assert any(m["type"] == "status" for m in output.messages)
    handle_frame(output.runtime, encode_frame(6, data[:, resumed : resumed + round(10.0 * SFREQ)]))
    after = observed[before_gap:]
    assert after[0].gap_before is True
    assert not any(o.gap_before for o in after[1:])
    assert after[0].decision_time_s == pytest.approx(10.0 + 6.0)


def test_an_out_of_order_frame_is_refused_without_changing_state(
    parts: SessionParts, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_logic_double(monkeypatch)
    runtime = handle_frame(start(parts), encode_frame(0, parts.probe.data[:, :FRAME])).runtime
    output = handle_frame(runtime, encode_frame(0, parts.probe.data[:, FRAME : 2 * FRAME]))
    assert output.runtime is runtime
    assert output.messages[0]["code"] == "out_of_order"


def test_a_terminal_state_closes_and_refuses_further_frames(
    parts: SessionParts, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_logic_double(monkeypatch, revoke_after=3)
    runtime, messages, transitions, closed = feed(start(parts), parts.probe.data)
    assert closed is True
    assert [t.to_state for t in transitions] == ["revoked"]
    assert sum(m["type"] == "window" for m in messages) == 3
    after = handle_frame(runtime, encode_frame(99, parts.probe.data[:, :FRAME]))
    assert after.close is True
    assert after.messages[0]["code"] == "session_ended"


def test_stop_records_a_closed_transition(
    parts: SessionParts, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_logic_double(monkeypatch)
    runtime = handle_frame(start(parts), encode_frame(0, parts.probe.data[:, :800])).runtime
    output = handle_stop(runtime)
    assert output.close is True
    assert [(t.from_state, t.to_state) for t in output.transitions] == [("active", "closed")]
    assert handle_stop(output.runtime).transitions == ()


def test_messages_never_carry_signal_features_or_template_data(
    parts: SessionParts, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_logic_double(monkeypatch, revoke_after=5)
    _, messages, _, _ = feed(start(parts), parts.probe.data[:, : round(15 * SFREQ)])
    assert messages
    for message in messages:
        for value in message.values():
            assert value is None or isinstance(value, bool | int | float | str)


def test_header_layout_is_pinned() -> None:
    assert FRAME_HEADER.format == "<4sIHH"
    assert FRAME_HEADER.size == struct.calcsize("<4sIHH") == 12
