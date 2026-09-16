"""Session state machine: confidence decay, challenge and revoke thresholds.

AUTHOR-SPECIFIED (CLAUDE.md lists this module on the author-writes-by-hand list). The types,
the signatures, the pre-registered parameters, and every rule the bodies implement — the skip
rule, the span bound, the gap rule, the order of judgement — are the author's, fixed before
the bodies existed. The bodies below were written to that specification on the author's
explicit instruction; the resolutions marked below were not in the specification and are the
implementer's, flagged for review.

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

**A run is bounded in time as well as in count** (pre-registered, D-025). Skipping decisions
lets a run span more than `revoke_dwell_decisions` seconds, so a run whose first and last
sub-threshold decisions are more than `revoke_dwell_max_span_s` apart expires: the counter
resets, and the decision that overran the bound starts a new run.

**A gap breaks a dwell run, independently of the span bound.** A run means consecutive
decisions on one continuous stream. After a gap the buffer was reset and warm-up restarted,
and the server cannot know how long the gap lasted: session time counts received samples, so
the span bound is blind to it. Carrying a run across a gap would let revocation fire on
evidence from before a disconnect of unknown length, so `gap_before` resets the counter.
The cost is recorded rather than hidden: an attacker who can drop frames can restart the
count, buying warm-up plus three decisions per gap. Closing that needs either an arrival
clock in the runtime or a gap budget, and both are parameters that would have to be
pre-registered (D-025). *Delegated to the implementer.* The other reading — that the span
bound already covers this, so a gap should leave the run alone — is defensible only if session
time tracked wall-clock time, and it does not.

**The levels are compared against the confidence, not the window score.** *Not specified;
implementer's resolution.* That is how the parameters were measured (the cost tables in D-025
run the EMA and count sub-threshold confidences), and it is what "recovery above 0.62"
already implies. Reading the levels against the raw window score instead is the other
defensible reading, and it would revoke far more often: the EMA exists to absorb single-window
dips, and the measured escape and t50 numbers would not describe the system.

**Unusable decisions do not move the confidence at all,** so signal quality cannot by itself
drive a challenge. Quality has its own path, `max_consecutive_not_ok`. *Not specified;
implementer's resolution.* The alternative reading — decaying the EMA toward a floor value on
a bad window, or decaying it toward nothing over elapsed time — would revoke genuine users for
a loose electrode through the identity path, which is exactly the confusion the skip rule was
pre-registered to avoid.

**min_scored_windows gates the identity transitions only** (challenge, recovery, revoke),
not expiry. *Not specified; implementer's resolution.* The gate exists because confidence is
meaningless before it has evidence, and expiry does not read confidence. Gating expiry too
would let a session that never produces a usable window sit in "active" indefinitely, which
is the stall the dwell bound exists to prevent.

Recovery is an ordinary transition from "challenged" to "active", with the reason naming the
level crossed. Revocation is the same shape with to_state "revoked"; because terminal states
absorb, the runtime then stops scoring, sends the transition, and closes the socket, and any
later frame is refused with "session_ended".
"""

import math
from dataclasses import dataclass, replace
from typing import Final

from neuroauth.verification.metrics import SessionStateName

TERMINAL_STATES: Final = frozenset({"revoked", "expired", "closed"})


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
        revoke_dwell_max_span_s: Seconds a dwell run may span from its first sub-threshold
            decision to its last. Past it the run expires and a new one begins (D-025).
        min_scored_windows: Scored windows required before any transition.
        max_consecutive_not_ok: Consecutive not-ok windows tolerated before the author's
            chosen consequence.
    """

    ema_half_life_s: float
    challenge_below: float
    revoke_below: float
    recover_above: float
    revoke_dwell_decisions: int
    revoke_dwell_max_span_s: float
    min_scored_windows: int
    max_consecutive_not_ok: int


PRE_REGISTERED_THRESHOLDS: Final = SessionThresholds(
    ema_half_life_s=4.0,
    challenge_below=0.58,
    revoke_below=0.56,
    recover_above=0.62,
    revoke_dwell_decisions=3,
    revoke_dwell_max_span_s=8.0,
    min_scored_windows=4,
    max_consecutive_not_ok=5,
)
"""Chosen from cohort scores and genuine-only cohort replays, never from the holdout (D-025).

The six pre-registered values are pinned by tests/test_session_parameters.py. min_scored_
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
        n_scored_windows: Usable windows (scored and quality-ok) so far.
        consecutive_not_ok: Current run of unusable windows, whether unscorable or flagged.
        last_decision_time_s: Time of the last window that moved the confidence, so the EMA
            decays over the gap between usable decisions. None before the first one.
        dwell_count: Sub-threshold decisions in the run in progress. A run cannot be
            rebuilt from confidence alone, so it is carried here.
        dwell_started_s: Decision time of that run's first sub-threshold decision, which the
            span bound measures from. None when no run is in progress.
    """

    state: SessionStateName
    confidence: float | None
    n_scored_windows: int
    consecutive_not_ok: int
    last_decision_time_s: float | None
    dwell_count: int = 0
    dwell_started_s: float | None = None


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
    """The state a session starts in: active, with no confidence and no prior.

    The EMA seeds from the first usable window rather than from an assumed value, so there
    is nothing to invent here. "active" means "not yet judged": the min_scored_windows gate
    keeps the session from transitioning until it has evidence, and a session that produces
    no usable window can still end through the quality path.
    """
    return SessionState(
        state="active",
        confidence=None,
        n_scored_windows=0,
        consecutive_not_ok=0,
        last_decision_time_s=None,
        dwell_count=0,
        dwell_started_s=None,
    )


def _usable(observation: WindowObservation) -> bool:
    """A decision the identity logic may read: scored, finite, and not quality-flagged."""
    return (
        observation.quality_ok
        and observation.score is not None
        and math.isfinite(observation.score)
    )


def _decayed(state: SessionState, score: float, now: float, half_life_s: float) -> float:
    """The EMA after one usable decision, seeding from the first one."""
    if state.confidence is None or state.last_decision_time_s is None:
        return score
    elapsed = now - state.last_decision_time_s
    if elapsed <= 0.0:
        # The runtime delivers strictly increasing decision times, so this is unreachable;
        # keeping the confidence is the conservative answer if that ever changes.
        return state.confidence
    alpha = 1.0 - float(0.5 ** (elapsed / half_life_s))
    return state.confidence + alpha * (score - state.confidence)


def _advance_dwell(
    state: SessionState, confidence: float, now: float, thresholds: SessionThresholds
) -> tuple[int, float | None]:
    """The dwell run after one usable decision: (count, run start)."""
    if confidence >= thresholds.revoke_below:
        return 0, None
    started = state.dwell_started_s
    if state.dwell_count == 0 or started is None:
        return 1, now
    if now - started > thresholds.revoke_dwell_max_span_s:
        return 1, now
    return state.dwell_count + 1, started


def update_session(
    state: SessionState,
    observation: WindowObservation,
    thresholds: SessionThresholds,
) -> tuple[SessionState, SessionTransition | None]:
    """Fold one window observation into the session.

    The order is: absorb terminal states; break any dwell run a gap interrupted; handle an
    unusable decision on the quality path and stop; otherwise decay the confidence, advance
    the dwell run, and only then judge. Judging goes revoke, challenge, recover, so the most
    severe outcome wins when one decision satisfies more than one.

    See the module docstring for the guarantees the runtime relies on, the transitions this
    may return, and the pre-registered rules it follows (D-025).

    Returns:
        (new_state, transition), with transition None unless state.state changed.
    """
    now = observation.decision_time_s
    if state.state in TERMINAL_STATES:
        return state, None

    carried = (0, None) if observation.gap_before else (state.dwell_count, state.dwell_started_s)
    state = replace(state, dwell_count=carried[0], dwell_started_s=carried[1])

    if not _usable(observation):
        consecutive = state.consecutive_not_ok + 1
        updated = replace(state, consecutive_not_ok=consecutive)
        if consecutive > thresholds.max_consecutive_not_ok:
            reason = (
                f"{consecutive} consecutive unusable windows, above the "
                f"{thresholds.max_consecutive_not_ok} tolerated"
            )
            return replace(updated, state="expired"), SessionTransition(
                state.state, "expired", now, state.confidence, reason
            )
        return updated, None

    score = float(observation.score) if observation.score is not None else 0.0
    confidence = _decayed(state, score, now, thresholds.ema_half_life_s)
    dwell_count, dwell_started = _advance_dwell(state, confidence, now, thresholds)
    scored = state.n_scored_windows + 1
    updated = replace(
        state,
        confidence=confidence,
        n_scored_windows=scored,
        consecutive_not_ok=0,
        last_decision_time_s=now,
        dwell_count=dwell_count,
        dwell_started_s=dwell_started,
    )
    if scored < thresholds.min_scored_windows:
        return updated, None

    def transition(
        to_state: SessionStateName, reason: str
    ) -> tuple[SessionState, SessionTransition]:
        return replace(updated, state=to_state), SessionTransition(
            state.state, to_state, now, confidence, reason
        )

    if dwell_count >= thresholds.revoke_dwell_decisions:
        return transition(
            "revoked",
            f"confidence {confidence:.3f} below {thresholds.revoke_below:g} for "
            f"{dwell_count} consecutive scored decisions",
        )
    if state.state == "active" and confidence < thresholds.challenge_below:
        return transition(
            "challenged",
            f"confidence {confidence:.3f} below the challenge level {thresholds.challenge_below:g}",
        )
    if state.state == "challenged" and confidence > thresholds.recover_above:
        return transition(
            "active",
            f"confidence {confidence:.3f} back above the recovery level "
            f"{thresholds.recover_above:g}",
        )
    return updated, None
