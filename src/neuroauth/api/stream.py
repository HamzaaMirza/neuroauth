"""WebSocket endpoint /stream/{session_id}.

The endpoint accepts the socket, resolves the session, and then shuttles messages through
neuroauth.session.runtime. Every behaviour worth testing lives in the runtime, which is
transport-free; this module only moves bytes and JSON (protocol in runtime's docstring).

Resolving a session (which verifier and thresholds apply) and recording transitions (the
events table, with provenance) are injected. The database arrives in Phase 2 item 5; until
then callers supply in-memory implementations.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from neuroauth.session.logic import SessionThresholds, SessionTransition
from neuroauth.session.runtime import (
    FrameError,
    RuntimeOutput,
    handle_frame,
    handle_stop,
    start_runtime,
)
from neuroauth.verification.scoring import Verifier

POLICY_VIOLATION = 1008


@dataclass(frozen=True, repr=False)
class SessionContext:
    """What a session needs at start: the claimed identity's verifier and the thresholds."""

    verifier: Verifier
    thresholds: SessionThresholds


class SessionResolver(Protocol):
    """Look up a session. Returns None for an unknown session or a mismatched claim."""

    def __call__(self, session_id: str, claimed_subject: str) -> SessionContext | None: ...


class TransitionSink(Protocol):
    """Record a session transition with provenance (events table in item 5)."""

    def __call__(
        self, session_id: str, transition: SessionTransition, context: SessionContext
    ) -> None: ...


def _parse_text(text: str) -> Mapping[str, object] | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def create_app(resolve: SessionResolver, record_transition: TransitionSink) -> FastAPI:
    """Build the FastAPI app with the streaming endpoint."""
    app = FastAPI(title="NeuroAuth")

    @app.websocket("/stream/{session_id}")
    async def stream(websocket: WebSocket, session_id: str) -> None:
        await websocket.accept()
        first = await websocket.receive()
        start = _parse_text(first["text"]) if first.get("text") is not None else None
        claimed = start.get("claimed_subject") if start is not None else None
        context = resolve(session_id, claimed) if isinstance(claimed, str) else None
        if start is None or context is None:
            refused = FrameError("bad_start", "unknown session or claimed subject")
            await websocket.send_json(refused.message())
            await websocket.close(code=POLICY_VIOLATION)
            return
        started = start_runtime(session_id, start, context.verifier, context.thresholds)
        if isinstance(started, FrameError):
            await websocket.send_json(started.message())
            await websocket.close(code=POLICY_VIOLATION)
            return
        runtime = started

        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    return
                output: RuntimeOutput
                if message.get("bytes") is not None:
                    output = handle_frame(runtime, message["bytes"])
                else:
                    control = _parse_text(message.get("text") or "")
                    if control is None or control.get("type") != "stop":
                        refused = FrameError("bad_message", "expected a binary frame or a stop")
                        await websocket.send_json(refused.message())
                        continue
                    output = handle_stop(runtime)
                runtime = output.runtime
                for transition in output.transitions:
                    record_transition(session_id, transition, context)
                for outbound in output.messages:
                    await websocket.send_json(dict(outbound))
                if output.close:
                    await websocket.close()
                    return
        except WebSocketDisconnect:
            return

    return app
