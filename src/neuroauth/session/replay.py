"""Replayed streams and injected impostors. SIMULATED, and labelled as such everywhere.

eegmmidb has no live headset, so every session is a recording replayed through the real
WebSocket path (sessions.stream_source = 'replay'). An impostor swap is two recordings
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
    raise NotImplementedError("TODO(phase-2): replay framing")


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
        (n_channels, n_samples) raw volts, ending when either source runs out.

    Raises:
        ValueError: If sfreq or ch_names differ, the swap or offset falls outside a
            recording, or second is first with second_offset_s equal to swap_at_s (a
            no-op splice).
    """
    raise NotImplementedError("TODO(phase-2): crossfaded raw splice")
