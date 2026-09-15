"""Session state machine: confidence decay, challenge and revoke thresholds.

AUTHOR-WRITTEN (CLAUDE.md). Only the types and signatures are scaffolded here. The field
lists on SessionThresholds and SessionState are a proposed shape; reshape them freely.

What the rest of the system relies on, and nothing more:

- update_session is pure and never raises, whatever the observation holds (score None,
  quality not ok, a gap).
- A SessionTransition is returned exactly when the state name changes, and never otherwise.
  The runtime writes each one to the events table with provenance: model_version,
  streaming fingerprint, thresholds, and actor.
- revoked, expired, and closed are absorbing.
- The runtime delivers observations in strictly increasing decision_time_s.
- The runtime reads only `state` and `confidence` from SessionState. On a client stop it
  records the close itself, by replacing `state` with "closed" (dataclasses.replace), since
  closing is a lifecycle event rather than a confidence decision. Every other field is the
  author's.

Where threshold values may come from: cohort-impostor scores and genuine-only replays of
enrollable subjects. Never impostor-holdout scores or holdout swap replays, which are
reserved for reporting time-to-detect (protocol.py).
"""

from dataclasses import dataclass

from neuroauth.verification.metrics import SessionStateName


@dataclass(frozen=True)
class SessionThresholds:
    """Proposed shape. Recorded verbatim in sessions.threshold_config.

    Attributes:
        ema_half_life_s: Decay of the confidence average, in seconds of decision time.
        challenge_below: Confidence below which an active session is challenged.
        revoke_below: Confidence below which a session is revoked.
        recover_above: Confidence above which a challenged session returns to active.
        min_scored_windows: Scored windows required before any transition.
        max_consecutive_not_ok: Consecutive not-ok windows tolerated before the author's
            chosen consequence.
    """

    ema_half_life_s: float
    challenge_below: float
    revoke_below: float
    recover_above: float
    min_scored_windows: int
    max_consecutive_not_ok: int


@dataclass(frozen=True)
class WindowObservation:
    """What the session logic sees for one window.

    Attributes:
        decision_time_s: WindowScore.decision_time_s.
        score: Hamming similarity in [0, 1], or None if unscorable.
        llr: The decision layer's log-likelihood ratio (D-021), or None if unscorable.
            Whether confidence is built from score or llr, and so the units of the
            thresholds, is the author's call.
        quality_ok: QualityReport.window_ok.
        gap_before: True if frames were lost since the previous observation and the stream
            buffer was reset.
    """

    decision_time_s: float
    score: float | None
    llr: float | None
    quality_ok: bool
    gap_before: bool


@dataclass(frozen=True)
class SessionState:
    """Proposed shape.

    Attributes:
        state: Matches the sessions.state CHECK in migrations/001_phase1_core.sql.
        confidence: None before the first scored window.
        n_scored_windows: Windows with a score so far.
        consecutive_not_ok: Current run of not-ok windows.
        last_decision_time_s: None before the first observation.
    """

    state: SessionStateName
    confidence: float | None
    n_scored_windows: int
    consecutive_not_ok: int
    last_decision_time_s: float | None


@dataclass(frozen=True)
class SessionTransition:
    """A state change, for the events table. The runtime adds provenance.

    Attributes:
        from_state: Before.
        to_state: After.
        decision_time_s: Of the observation that caused it.
        confidence: After the update.
        reason: Human-readable, e.g. "confidence 0.41 below revoke threshold 0.45".
    """

    from_state: SessionStateName
    to_state: SessionStateName
    decision_time_s: float
    confidence: float | None
    reason: str


def initial_session_state() -> SessionState:
    """The state a session starts in.

    Author's. The starting confidence is a prior, and the prior is part of the confidence
    logic.
    """
    raise NotImplementedError("TODO(author): initial session state and confidence prior")


def update_session(
    state: SessionState,
    observation: WindowObservation,
    thresholds: SessionThresholds,
) -> tuple[SessionState, SessionTransition | None]:
    """Fold one window observation into the session.

    Author's. See the module docstring for the guarantees the runtime relies on.

    Returns:
        (new_state, transition), with transition None unless state.state changed.
    """
    raise NotImplementedError("TODO(author): EMA confidence decay, challenge/revoke logic")
