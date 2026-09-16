"""Session state machine: confidence decay, challenge and revoke thresholds.

AUTHOR-WRITTEN (CLAUDE.md). Only the types, the signatures, and the pre-registered parameter
values are settled here. The bodies of `update_session` and `initial_session_state` are the
author's.

**The unit is the protected score**, WindowObservation.score: Hamming similarity in [0, 1],
in steps of 1/64. Not the LLR. A global LLR threshold gives the weakest accounts a higher FAR
than the rest, which hard rule 4 forbids (D-025). The LLR travels beside it for logging and
for Phase 3, and nothing in the session logic reads it.

**Pre-registered parameters** (D-025, fixed 2026-09-16 from cohort data only, before any
holdout session result): PRE_REGISTERED_THRESHOLDS below. Under the D-016 rule they are not
adjustable after holdout results; a change needs a DECISIONS entry reporting both values.

What the rest of the system relies on, and nothing more:

- update_session is pure and never raises, whatever the observation holds (score None,
  quality not ok, a gap).
- A SessionTransition is returned exactly when the state name changes, and never otherwise.
  The runtime writes each one to the events table with provenance: representation_version,
  decision_version, streaming fingerprint, thresholds, and actor.
- revoked, expired, and closed are absorbing.
- The runtime delivers observations in strictly increasing decision_time_s.
- The runtime reads only `state` and `confidence` from SessionState. On a client stop it
  records the close itself, by replacing `state` with "closed" (dataclasses.replace), since
  closing is a lifecycle event rather than a confidence decision. Every other field is the
  author's.

**Which transitions update_session may return** (the runtime owns "closed" only):

| From | To | Meaning |
|---|---|---|
| active | challenged | confidence fell below challenge_below: step up, not a lockout |
| challenged | active | recovery: confidence climbed back to recover_above |
| active or challenged | revoked | confidence below revoke_below for the whole dwell. Terminal |
| active or challenged | expired | the author's timeout, if one is implemented. Terminal |

**Revocation carries a dwell, challenge does not** (D-025). Revoking takes
`revoke_dwell_decisions` consecutive decisions below `revoke_below`; a single decision below
`challenge_below` challenges immediately, because a spurious step-up prompt is cheap and a
dwell would only delay it.

**The dwell counts scored, quality-ok decisions only** (pre-registered, D-025). A decision
with no score, or one the quality mask flagged, neither advances nor resets the run: it is
skipped. Resetting on a bad window would let an impostor stall revocation by inducing bad
signal; counting one as sub-threshold would revoke genuine users for a hardware fault.
Sustained bad signal is not ignored, it is `max_consecutive_not_ok`'s path, so a session
still ends, through a route that says the signal went bad rather than that the person
changed.

Recovery is an ordinary transition from "challenged" to "active", with the reason naming the
level crossed. Revocation is the same shape with to_state "revoked"; because terminal states
absorb, the runtime then stops scoring, sends the transition, and closes the socket, and any
later frame is refused with "session_ended".
"""

from dataclasses import dataclass
from typing import Final

from neuroauth.verification.metrics import SessionStateName


@dataclass(frozen=True)
class SessionThresholds:
    """Session parameters, recorded verbatim in sessions.threshold_config.

    The first five are pre-registered (D-025) and fixed under the D-016 rule. The last two
    are the author's and may change without a DECISIONS entry, because they judge no result.

    Attributes:
        ema_half_life_s: Decay of the confidence average, in seconds of decision time.
        challenge_below: Confidence below which an active session is challenged, on the
            first decision. No dwell.
        revoke_below: Confidence below which a session is revoked, once the dwell is met.
            Terminal.
        recover_above: Confidence above which a challenged session returns to active.
        revoke_dwell_decisions: Consecutive decisions below revoke_below needed to revoke.
            1 would revoke on the first. Counts scored, quality-ok decisions only;
            unscorable or flagged ones are skipped (D-025).
        min_scored_windows: Scored windows required before any transition.
        max_consecutive_not_ok: Consecutive not-ok windows tolerated before the author's
            chosen consequence.
    """

    ema_half_life_s: float
    challenge_below: float
    revoke_below: float
    recover_above: float
    revoke_dwell_decisions: int
    min_scored_windows: int
    max_consecutive_not_ok: int


PRE_REGISTERED_THRESHOLDS: Final = SessionThresholds(
    ema_half_life_s=4.0,
    challenge_below=0.58,
    revoke_below=0.56,
    recover_above=0.62,
    revoke_dwell_decisions=3,
    min_scored_windows=4,
    max_consecutive_not_ok=5,
)
"""Chosen from cohort scores and genuine-only cohort replays, never from the holdout (D-025).

The five pre-registered values are pinned by tests/test_session_parameters.py. min_scored_
windows is one half-life of decisions at the 1 s hop; it and max_consecutive_not_ok are the
author's."""


@dataclass(frozen=True)
class WindowObservation:
    """What the session logic sees for one window.

    Attributes:
        decision_time_s: WindowScore.decision_time_s.
        score: Hamming similarity in [0, 1], or None if unscorable. This is the unit the
            thresholds are in.
        llr: The decision layer's log-likelihood ratio (D-021), or None if unscorable.
            Carried for logging and Phase 3; the session logic does not read it (D-025).
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
    """The session as the logic sees it. The runtime reads only `state` and `confidence`.

    Attributes:
        state: Matches the sessions.state CHECK in migrations/001_phase1_core.sql.
        confidence: The EMA of the protected score, in the same units as the thresholds.
            None before the first scored window, which is what min_scored_windows gates on.
        n_scored_windows: Windows with a score so far.
        consecutive_not_ok: Current run of not-ok windows.
        last_decision_time_s: None before the first observation. The EMA uses the gap to it.
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
        reason: Human-readable, e.g. "confidence 0.541 below revoke threshold 0.56".
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

    Author's. See the module docstring for the guarantees the runtime relies on and for the
    transitions this may return.

    Returns:
        (new_state, transition), with transition None unless state.state changed.
    """
    raise NotImplementedError("TODO(author): EMA confidence decay, challenge/revoke logic")
