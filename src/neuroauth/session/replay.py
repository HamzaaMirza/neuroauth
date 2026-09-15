"""Replayed streams and injected impostors. SIMULATED, and labelled as such everywhere.

eegmmidb has no live headset, so every session is a recording replayed through the real
frame protocol (sessions.stream_source = 'replay'). An impostor swap is two recordings
spliced together. That splice is the simulated part most likely to flatter the system: a
hard cut is a step in 64 channels, a broadband transient that could trigger a quality flag or
an odd spectrum and look like detection. Two defences, both reported: a raised-cosine
crossfade, and a self-splice control judged by metrics.SPLICE_CONTROL_MAX_EXCESS.
"""

from collections.abc import Iterator
from typing import Final

import numpy as np
from numpy.typing import NDArray

from neuroauth.dsp.types import Recording
from neuroauth.session.logic import SessionThresholds
from neuroauth.session.runtime import (
    FrameError,
    encode_frame,
    handle_frame,
    handle_stop,
    start_runtime,
)
from neuroauth.verification.metrics import SessionStateName, SessionTrace
from neuroauth.verification.scoring import Verifier

SWAP_AT_S: Final = 30.0
"""Session time at which the crossfade to the impostor begins. Leaves 24 s of scored genuine
signal after the 6 s warm-up."""

CROSSFADE_S: Final = 0.5

DETECTION_HORIZON_S: Final = 25.0
"""Time after the swap within which detection counts. Bounded by recording length minus the
right margin."""

REPLAY_CHUNK_S: Final = 0.1
"""Frame duration for replay. All four constants fixed before any Phase 2 result."""


def replay_chunks(
    data: NDArray[np.float64], sfreq: float, *, chunk_s: float = REPLAY_CHUNK_S
) -> Iterator[NDArray[np.float64]]:
    """Cut (n_channels, n_samples) into consecutive frames of chunk_s. The last may be short.

    No real-time pacing. The replay client sleeps between frames; offline evaluation does
    not.

    Raises:
        ValueError: If data is not 2-D or chunk_s is below one sample.
    """
    x = np.asarray(data, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected (n_channels, n_samples), got shape {x.shape}")
    size = round(chunk_s * sfreq)
    if size < 1:
        raise ValueError(f"chunk_s {chunk_s} is below one sample at {sfreq} Hz")
    for start in range(0, x.shape[1], size):
        yield x[:, start : start + size]


def splice(
    first: Recording,
    second: Recording,
    *,
    swap_at_s: float = SWAP_AT_S,
    crossfade_s: float = CROSSFADE_S,
    second_offset_s: float | None = None,
) -> NDArray[np.float64]:
    """Raw-domain splice: first up to swap_at_s, raised-cosine crossfade, then second.

    The splice happens before any filtering, as it would on a real electrode swap, so the
    stream buffer sees what a live system would.

    Used for impostor swaps, where second is another subject's probe recording read from
    swap_at_s so both sources are at a comparable point, and for the self-splice control,
    where second is the same probe recording read from a different second_offset_s. The
    self-splice keeps the discontinuity and removes the identity change.

    Args:
        first: Supplies samples before the swap.
        second: Supplies samples after it.
        swap_at_s: Where the crossfade begins, in first's time.
        crossfade_s: Crossfade length. 0 gives a hard cut, which is used only to report
            how much the crossfade matters.
        second_offset_s: Where in second the post-swap signal is read from. None means
            swap_at_s.

    Returns:
        (n_channels, n_samples) raw volts: first's samples before the swap, the crossfade,
        then second's samples until second runs out.

    Raises:
        ValueError: If sfreq or ch_names differ, the swap or offset falls outside a
            recording, or second is first with second_offset_s equal to swap_at_s (a
            no-op splice).
    """
    if first.sfreq != second.sfreq or first.ch_names != second.ch_names:
        raise ValueError("spliced recordings must share sfreq and channel order")
    offset_s = swap_at_s if second_offset_s is None else second_offset_s
    same_recording = (first.subject_id, first.run) == (second.subject_id, second.run)
    if same_recording and offset_s == swap_at_s:
        raise ValueError("splicing a recording onto itself at the same offset changes nothing")
    swap = round(swap_at_s * first.sfreq)
    fade = round(crossfade_s * first.sfreq)
    offset = round(offset_s * first.sfreq)
    if swap < 0 or fade < 0 or swap + fade > first.data.shape[1]:
        raise ValueError("swap and crossfade fall outside the first recording")
    if offset < 0 or offset + fade >= second.data.shape[1]:
        raise ValueError("offset and crossfade fall outside the second recording")

    weight = 0.5 - 0.5 * np.cos(np.pi * (np.arange(fade) + 0.5) / fade) if fade else np.empty(0)
    blended = (
        first.data[:, swap : swap + fade] * (1.0 - weight)
        + second.data[:, offset : offset + fade] * weight
    )
    return np.concatenate(
        (first.data[:, :swap], blended, second.data[:, offset + fade :]), axis=1
    ).astype(np.float64)


def replay_session(
    data: NDArray[np.float64],
    *,
    verifier: Verifier,
    thresholds: SessionThresholds,
    claimed_subject: int,
    ch_names: tuple[str, ...],
    sfreq: float,
    swap_at_s: float | None,
    impostor_subject: int | None,
    chunk_s: float = REPLAY_CHUNK_S,
    session_id: str = "replay",
) -> SessionTrace:
    """Replay a stream through the real frame protocol and session runtime.

    Every frame is encoded and decoded exactly as over the WebSocket, so offline replays
    and the live endpoint exercise one path. Stops at the first terminal state.

    Raises:
        ValueError: If the runtime refuses the start message.
    """
    started = start_runtime(
        session_id,
        {
            "type": "start",
            "claimed_subject": verifier.template.subject_ref,
            "sfreq": sfreq,
            "ch_names": list(ch_names),
            "stream_source": "replay",
        },
        verifier,
        thresholds,
    )
    if isinstance(started, FrameError):
        raise ValueError(f"replay refused at start: {started.code}: {started.detail}")
    runtime = started
    times: list[float] = []
    states: list[SessionStateName] = []
    for seq, chunk in enumerate(replay_chunks(data, sfreq, chunk_s=chunk_s)):
        output = handle_frame(runtime, encode_frame(seq, chunk))
        runtime = output.runtime
        for message in output.messages:
            if message.get("type") == "window":
                times.append(float(message["decision_time_s"]))  # type: ignore[arg-type]
                states.append(message["state"])  # type: ignore[arg-type]
        if output.close:
            break
    else:
        handle_stop(runtime)
    return SessionTrace(
        claimed_subject=claimed_subject,
        decision_times_s=np.asarray(times, dtype=np.float64),
        states=tuple(states),
        swap_at_s=swap_at_s,
        impostor_subject=impostor_subject,
    )
