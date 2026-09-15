"""The WebSocket session, minus the socket: frame decoding, buffering, scoring, state.

Wire protocol for /stream/{session_id}:

Client to server:
    text   {"type": "start", "claimed_subject": "eegmmidb-S001", "sfreq": 160.0,
            "ch_names": [...64 names...], "stream_source": "replay"}
    binary frame = header + payload
           header  struct "<4sIHH": magic b"NAF1", seq (uint32, from 0, +1 per frame),
                   n_channels (uint16), n_samples (uint16)
           payload float64 little-endian, C order, shape (n_channels, n_samples), volts
    text   {"type": "stop"}

Server to client (JSON text):
    {"type": "status", "state", "warming_up": bool, "ready_in_s": float}
    {"type": "window", "onset_s", "decision_time_s", "score", "llr", "quality_ok", "state",
     "confidence"}
    {"type": "transition", "from", "to", "decision_time_s", "reason"}
    {"type": "error", "code", "detail"}

Never in any message: samples, features, embeddings, template bits, key material.

The payload is float64, not float32. float32 rounds at about 1e-7 relative, which would break
the bit-for-bit identity between replayed and offline features that the edge-effect tests
assert end to end. At 160 Hz and 64 channels, float64 is 82 kB/s.

Gaps: a frame whose seq skips ahead means lost samples. The stream buffer is reset rather
than filtered across the hole, the next observation carries gap_before=True, and warm-up
restarts. Session time counts received samples; the duration of a gap is unknown to the
server and is not counted. A frame whose seq is behind the expected one is a duplicate or
out of order, and is refused without changing state.

Session logic (update_session, initial_session_state) is the author's and is called through
the logic module, so the runtime always uses the current implementation.
"""

import struct
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray

from neuroauth.dsp.features import feature_channels
from neuroauth.dsp.streaming import (
    StreamBuffer,
    new_stream_buffer,
    push_samples,
    samples_until_next_window,
    segment_warming_up,
)
from neuroauth.session import logic
from neuroauth.session.logic import (
    SessionState,
    SessionThresholds,
    SessionTransition,
    WindowObservation,
)
from neuroauth.verification.scoring import Verifier, score_windows

FRAME_MAGIC: Final = b"NAF1"
FRAME_HEADER: Final = struct.Struct("<4sIHH")
MAX_FRAME_SAMPLES: Final = 1600
"""10 s at 160 Hz. Larger frames are refused, which bounds per-frame work and memory."""

EXPECTED_SFREQ: Final = 160.0
"""eegmmidb's rate, and the rate every Phase 2 model's features were built at."""

TERMINAL_STATES: Final = frozenset({"revoked", "expired", "closed"})

FrameErrorCode = Literal[
    "bad_magic",
    "short_header",
    "length_mismatch",
    "channel_mismatch",
    "too_many_samples",
    "bad_start",
    "bad_message",
    "out_of_order",
    "session_ended",
]


@dataclass(frozen=True)
class SampleFrame:
    """A decoded frame. samples is (n_channels, n_samples) float64 volts."""

    seq: int
    samples: NDArray[np.float64]


@dataclass(frozen=True)
class FrameError:
    """A frame that could not be used. Reported to the client, never raised."""

    code: FrameErrorCode
    detail: str

    def message(self) -> dict[str, object]:
        return {"type": "error", "code": self.code, "detail": self.detail}


def decode_frame(payload: bytes, n_channels: int) -> SampleFrame | FrameError:
    """Parse one binary frame. Never raises.

    Non-finite samples are not an error here. They pass through to the quality mask, as
    everywhere else in the pipeline.
    """
    if len(payload) < FRAME_HEADER.size:
        return FrameError("short_header", f"frame of {len(payload)} bytes has no full header")
    magic, seq, channels, samples = FRAME_HEADER.unpack_from(payload)
    if magic != FRAME_MAGIC:
        return FrameError("bad_magic", "frame does not start with NAF1")
    if channels != n_channels:
        return FrameError(
            "channel_mismatch", f"frame has {channels} channels, expected {n_channels}"
        )
    if samples > MAX_FRAME_SAMPLES:
        return FrameError(
            "too_many_samples", f"frame has {samples} samples, at most {MAX_FRAME_SAMPLES}"
        )
    expected = FRAME_HEADER.size + 8 * channels * samples
    if len(payload) != expected:
        return FrameError("length_mismatch", f"frame is {len(payload)} bytes, expected {expected}")
    data = np.frombuffer(payload, dtype="<f8", offset=FRAME_HEADER.size)
    return SampleFrame(seq=int(seq), samples=data.reshape(channels, samples).astype(np.float64))


def encode_frame(seq: int, samples: NDArray[np.float64]) -> bytes:
    """Build one binary frame. Used by the replay client and by tests.

    Raises:
        ValueError: If samples is not 2-D, is too large, or seq is out of range. This is
            the client side, not the inference path.
    """
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected (n_channels, n_samples), got shape {x.shape}")
    if not 1 <= x.shape[0] <= 0xFFFF or x.shape[1] > MAX_FRAME_SAMPLES:
        raise ValueError(f"frame shape {x.shape} is outside the protocol's limits")
    if not 0 <= seq <= 0xFFFFFFFF:
        raise ValueError(f"seq {seq} does not fit in uint32")
    header = FRAME_HEADER.pack(FRAME_MAGIC, seq, x.shape[0], x.shape[1])
    return header + np.ascontiguousarray(x, dtype="<f8").tobytes()


@dataclass(frozen=True, repr=False)
class SessionRuntime:
    """All in-memory state of one live session. Never persisted as a whole.

    repr=False because it holds raw samples and a key-derived projection.

    Attributes:
        session_id: sessions.id.
        verifier: Claimed identity's verifier.
        thresholds: In effect for this session, recorded at start.
        buffer: Raw samples awaiting context.
        session: update_session state.
        sfreq: Declared at start; must be EXPECTED_SFREQ.
        ch_names: Declared at start; must equal the model's channel order.
        expected_seq: Next frame sequence number.
        samples_received: Total samples accepted, for session time.
        pending_gap: A gap happened and no observation has carried the flag yet.
    """

    session_id: str
    verifier: Verifier
    thresholds: SessionThresholds
    buffer: StreamBuffer
    session: SessionState
    sfreq: float
    ch_names: tuple[str, ...]
    expected_seq: int
    samples_received: int
    pending_gap: bool


@dataclass(frozen=True)
class RuntimeOutput:
    """Result of handling one inbound message.

    Attributes:
        runtime: Advanced state.
        messages: JSON-serializable dicts to send, in order.
        transitions: For the events table, in order.
        close: True once the session is in a terminal state and the socket should close.
    """

    runtime: SessionRuntime
    messages: tuple[Mapping[str, object], ...]
    transitions: tuple[SessionTransition, ...]
    close: bool


def start_runtime(
    session_id: str,
    start_message: Mapping[str, object],
    verifier: Verifier,
    thresholds: SessionThresholds,
) -> SessionRuntime | FrameError:
    """Validate a start message against the verifier's configuration and open the runtime.

    Returns FrameError "bad_start" for a malformed message or a claimed subject other than
    the verifier's, and "channel_mismatch" when sfreq or ch_names disagree with the model's
    pipeline. Never raises on the message.
    """
    sfreq = start_message.get("sfreq")
    ch_names = start_message.get("ch_names")
    if (
        start_message.get("type") != "start"
        or isinstance(sfreq, bool)
        or not isinstance(sfreq, int | float)
        or not isinstance(ch_names, list)
        or not all(isinstance(name, str) for name in ch_names)
    ):
        return FrameError("bad_start", "expected a start message with sfreq and ch_names")
    if start_message.get("claimed_subject") != verifier.template.subject_ref:
        return FrameError("bad_start", "claimed_subject does not match the session's template")
    expected_channels = feature_channels(verifier.model.feature_names)
    if float(sfreq) != EXPECTED_SFREQ or tuple(ch_names) != expected_channels:
        return FrameError(
            "channel_mismatch",
            f"stream must be {EXPECTED_SFREQ} Hz with the model's {len(expected_channels)} "
            "channels in order",
        )
    return SessionRuntime(
        session_id=session_id,
        verifier=verifier,
        thresholds=thresholds,
        buffer=new_stream_buffer(len(ch_names), verifier.streaming, float(sfreq)),
        session=logic.initial_session_state(),
        sfreq=float(sfreq),
        ch_names=tuple(ch_names),
        expected_seq=0,
        samples_received=0,
        pending_gap=False,
    )


def _window_message(
    score_onset: float, observation: WindowObservation, state: SessionState
) -> dict[str, object]:
    return {
        "type": "window",
        "onset_s": score_onset,
        "decision_time_s": observation.decision_time_s,
        "score": observation.score,
        "llr": observation.llr,
        "quality_ok": observation.quality_ok,
        "state": state.state,
        "confidence": state.confidence,
    }


def _transition_message(transition: SessionTransition) -> dict[str, object]:
    return {
        "type": "transition",
        "from": transition.from_state,
        "to": transition.to_state,
        "decision_time_s": transition.decision_time_s,
        "reason": transition.reason,
    }


def _status_message(runtime: SessionRuntime) -> dict[str, object]:
    missing = samples_until_next_window(runtime.buffer, runtime.sfreq, runtime.verifier.streaming)
    return {
        "type": "status",
        "state": runtime.session.state,
        "warming_up": True,
        "ready_in_s": missing / runtime.sfreq,
    }


def handle_frame(runtime: SessionRuntime, payload: bytes) -> RuntimeOutput:
    """Process one binary frame end to end. Never raises on the frame.

    decode, then check seq (on a gap, reset the buffer and flag the next observation), then
    push_samples, score_windows, and update_session once per window in decision-time order,
    then build messages. After a terminal state, frames are refused with "session_ended" and
    close=True. Windows emitted after a transition into a terminal state are not scored.
    """
    if runtime.session.state in TERMINAL_STATES:
        ended = FrameError("session_ended", f"session is {runtime.session.state}")
        return RuntimeOutput(runtime, (ended.message(),), (), True)
    decoded = decode_frame(payload, len(runtime.ch_names))
    if isinstance(decoded, FrameError):
        return RuntimeOutput(runtime, (decoded.message(),), (), False)
    if decoded.seq < runtime.expected_seq:
        stale = FrameError(
            "out_of_order", f"frame seq {decoded.seq} is behind expected {runtime.expected_seq}"
        )
        return RuntimeOutput(runtime, (stale.message(),), (), False)

    streaming = runtime.verifier.streaming
    buffer = runtime.buffer
    pending_gap = runtime.pending_gap
    if decoded.seq > runtime.expected_seq:
        buffer = new_stream_buffer(
            len(runtime.ch_names),
            streaming,
            runtime.sfreq,
            origin_s=runtime.samples_received / runtime.sfreq,
        )
        pending_gap = True

    buffer, features = push_samples(
        buffer, decoded.samples, runtime.sfreq, runtime.ch_names, streaming
    )
    session = runtime.session
    messages: list[Mapping[str, object]] = []
    transitions: list[SessionTransition] = []
    for window in score_windows(runtime.verifier, features):
        observation = WindowObservation(
            decision_time_s=window.decision_time_s,
            score=window.score,
            llr=window.llr,
            quality_ok=window.quality_ok,
            gap_before=pending_gap,
        )
        pending_gap = False
        session, transition = logic.update_session(session, observation, runtime.thresholds)
        messages.append(_window_message(window.onset_s, observation, session))
        if transition is not None:
            transitions.append(transition)
            messages.append(_transition_message(transition))
        if session.state in TERMINAL_STATES:
            break

    advanced = replace(
        runtime,
        buffer=buffer,
        session=session,
        expected_seq=decoded.seq + 1,
        samples_received=runtime.samples_received + int(decoded.samples.shape[1]),
        pending_gap=pending_gap,
    )
    if features.values.shape[0] == 0 and segment_warming_up(buffer, runtime.sfreq, streaming):
        messages.append(_status_message(advanced))
    return RuntimeOutput(
        advanced, tuple(messages), tuple(transitions), session.state in TERMINAL_STATES
    )


def handle_stop(
    runtime: SessionRuntime, *, reason: str = "client stopped the stream"
) -> RuntimeOutput:
    """Close a session at the client's request.

    Closing is a lifecycle event, not a confidence decision, so the runtime records it: a
    transition to "closed" unless the session is already terminal.
    """
    if runtime.session.state in TERMINAL_STATES:
        return RuntimeOutput(runtime, (), (), True)
    transition = SessionTransition(
        from_state=runtime.session.state,
        to_state="closed",
        decision_time_s=runtime.samples_received / runtime.sfreq,
        confidence=runtime.session.confidence,
        reason=reason,
    )
    closed = replace(runtime, session=replace(runtime.session, state="closed"))
    return RuntimeOutput(closed, (_transition_message(transition),), (transition,), True)
