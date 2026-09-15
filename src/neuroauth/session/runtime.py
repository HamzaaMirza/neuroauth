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
    {"type": "status", "state": ..., "warming_up": bool, "ready_in_s": float}
    {"type": "window", "onset_s", "decision_time_s", "score", "quality_ok", "state",
     "confidence"}
    {"type": "transition", "from", "to", "decision_time_s", "reason"}
    {"type": "error", "code", "detail"}

Never in any message: samples, features, embeddings, template bits, key material.

The payload is float64, not float32. float32 rounds at about 1e-7 relative, which would break
the bit-for-bit identity between replayed and offline features that the edge-effect tests
assert end to end. At 160 Hz and 64 channels, float64 is 82 kB/s.

Gaps: a frame whose seq is not the expected one means lost samples. The stream buffer is
reset rather than filtered across the hole, the next observation carries gap_before=True,
and warm-up restarts.
"""

import struct
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray

from neuroauth.dsp.streaming import StreamBuffer
from neuroauth.session.logic import SessionState, SessionThresholds, SessionTransition
from neuroauth.verification.scoring import Verifier

FRAME_MAGIC: Final = b"NAF1"
FRAME_HEADER: Final = struct.Struct("<4sIHH")
MAX_FRAME_SAMPLES: Final = 1600
"""10 s at 160 Hz. Larger frames are refused, which bounds per-frame work and memory."""

FrameErrorCode = Literal[
    "bad_magic",
    "short_header",
    "length_mismatch",
    "channel_mismatch",
    "too_many_samples",
    "not_started",
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


def decode_frame(payload: bytes, n_channels: int) -> SampleFrame | FrameError:
    """Parse one binary frame. Never raises.

    Non-finite samples are not an error here. They pass through to the quality mask, as
    everywhere else in the pipeline.
    """
    raise NotImplementedError("TODO(phase-2): frame decoding")


def encode_frame(seq: int, samples: NDArray[np.float64]) -> bytes:
    """Build one binary frame. Used by the replay client and by tests.

    Raises:
        ValueError: If samples is not 2-D, is too large, or seq is out of range. This is
            the client side, not the inference path.
    """
    raise NotImplementedError("TODO(phase-2): frame encoding")


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
        sfreq: Declared at start; must match the model's pipeline.
        ch_names: Declared at start; must equal the model's channel order.
        expected_seq: Next frame sequence number.
        samples_received: Total samples accepted, for session time.
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

    Returns FrameError (code "channel_mismatch") when sfreq or ch_names disagree with the
    model's pipeline. Never raises.
    """
    raise NotImplementedError("TODO(phase-2): runtime start")


def handle_frame(runtime: SessionRuntime, payload: bytes) -> RuntimeOutput:
    """Process one binary frame end to end. Never raises.

    decode, then check seq (on a gap, reset the buffer and flag the next observation), then
    push_samples, score_windows, and update_session once per window in decision-time order,
    then build messages. After a terminal state, frames are ignored with a "session_ended"
    error and close=True.
    """
    raise NotImplementedError("TODO(phase-2): frame -> windows -> scores -> state")
