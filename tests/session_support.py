"""Shared parts for session tests: a small real verifier over synthetic streams, and a test
double for the author's session logic.

The double is not session logic. It exists so the runtime can be exercised before the
author writes update_session and initial_session_state, and it is installed with
monkeypatch so the real stubs stay untouched.
"""

import dataclasses
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pytest

from neuroauth.config import StreamingConfig
from neuroauth.dsp.streaming import process_recording_bounded
from neuroauth.dsp.types import Condition, Recording
from neuroauth.session import logic
from neuroauth.session.logic import (
    PRE_REGISTERED_THRESHOLDS,
    SessionState,
    SessionThresholds,
    SessionTransition,
    WindowObservation,
)
from neuroauth.templates.enrollment import enroll_subject, representation_version, subject_ref
from neuroauth.verification.decision import fit_decision_layer
from neuroauth.verification.embedding import EmbeddingConfig, fit_embedding
from neuroauth.verification.metrics import ScoreTable
from neuroauth.verification.scoring import Verifier, load_verifier
from tests.synthetic import CH_NAMES, SFREQ, eeg_like_data

STREAMING = StreamingConfig()
CHANNELS = CH_NAMES[:8]
SECRET = bytes(range(32))
HOLDOUT = frozenset({90})
THRESHOLDS = PRE_REGISTERED_THRESHOLDS
"""The session double below ignores these; tests use the registered values so a parameter
change shows up here too."""


def recording(subject: int, run: int, *, seed: int, duration_s: float = 40.0) -> Recording:
    condition: Condition = "eyes_open" if run == 1 else "eyes_closed"
    return Recording(
        data=eeg_like_data(seed=seed, duration_s=duration_s),
        sfreq=SFREQ,
        ch_names=CHANNELS,
        subject_id=subject,
        run=run,
        condition=condition,
    )


@dataclass(frozen=True, repr=False)
class SessionParts:
    verifier: Verifier
    probe: Recording
    impostor: Recording


def build_session_parts() -> SessionParts:
    """A verifier for subject 5, fitted on subjects 1-4, over bounded-context features."""
    training = [
        process_recording_bounded(recording(s, run, seed=10 * s + run, duration_s=30.0), STREAMING)
        for s in range(1, 5)
        for run in (1, 2)
    ]
    model = fit_embedding(
        training,
        config=EmbeddingConfig(),
        streaming_fingerprint=STREAMING.fingerprint(),
        impostor_holdout=HOLDOUT,
        evaluated_subjects=frozenset({5, 6}),
    )
    template = enroll_subject(
        process_recording_bounded(recording(5, 1, seed=51), STREAMING),
        model,
        SECRET,
        subject_id=5,
        subject_ref=subject_ref(5),
        impostor_holdout=HOLDOUT,
        now=datetime(2026, 9, 15, tzinfo=UTC),
    )
    rng = np.random.default_rng(0)
    n_rows = 200
    decision = fit_decision_layer(
        ScoreTable(
            scores=np.concatenate((rng.normal(0.8, 0.05, 100), rng.normal(0.5, 0.05, 100))),
            claimed_subject=np.full(n_rows, 5, dtype=np.int64),
            source_subject=np.repeat(np.array([5, 6], dtype=np.int64), 100),
            source_is_holdout=np.zeros(n_rows, dtype=np.bool_),
            fold=np.zeros(n_rows, dtype=np.int64),
            probe_onset_s=np.arange(n_rows, dtype=np.float64),
            probe_window_ok=np.ones(n_rows, dtype=np.bool_),
            template_score_mean=np.full(n_rows, 0.85),
            template_score_std=np.full(n_rows, 0.05),
            domain="protected",
            split_kind="cross_condition",
        ),
        n_bits=model.n_components,
        representation_versions=(representation_version(model),),
    )
    return SessionParts(
        verifier=load_verifier(template, model, decision, SECRET, STREAMING),
        probe=recording(5, 2, seed=52),
        impostor=recording(6, 2, seed=62),
    )


def start_message(parts: SessionParts) -> dict[str, object]:
    return {
        "type": "start",
        "claimed_subject": parts.verifier.template.subject_ref,
        "sfreq": SFREQ,
        "ch_names": list(CHANNELS),
        "stream_source": "replay",
    }


def install_logic_double(
    monkeypatch: pytest.MonkeyPatch, *, revoke_after: int | None = None
) -> list[WindowObservation]:
    """Replace the author's stubs for one test. Returns the observations it receives.

    Confidence is just the latest score; the session revokes after revoke_after scored
    windows when that is set.
    """
    observed: list[WindowObservation] = []

    def initial_session_state() -> SessionState:
        return SessionState("active", None, 0, 0, None)

    def update_session(
        state: SessionState, observation: WindowObservation, thresholds: SessionThresholds
    ) -> tuple[SessionState, SessionTransition | None]:
        observed.append(observation)
        if state.state in ("revoked", "expired", "closed"):
            return state, None
        scored = state.n_scored_windows + int(observation.score is not None)
        new = SessionState(state.state, observation.score, scored, 0, observation.decision_time_s)
        if revoke_after is not None and scored >= revoke_after:
            revoked = dataclasses.replace(new, state="revoked")
            transition = SessionTransition(
                state.state, "revoked", observation.decision_time_s, observation.score, "double"
            )
            return revoked, transition
        return new, None

    monkeypatch.setattr(logic, "initial_session_state", initial_session_state)
    monkeypatch.setattr(logic, "update_session", update_session)
    return observed
