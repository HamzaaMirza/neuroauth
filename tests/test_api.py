"""The WebSocket endpoint moves bytes through the runtime and records transitions."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from neuroauth.api.stream import SessionContext, create_app
from neuroauth.session.logic import SessionTransition
from neuroauth.session.runtime import encode_frame
from tests.session_support import (
    THRESHOLDS,
    SessionParts,
    build_session_parts,
    install_logic_double,
    start_message,
)
from tests.synthetic import SFREQ


@pytest.fixture(scope="module")
def parts() -> SessionParts:
    return build_session_parts()


def client(parts: SessionParts, recorded: list[tuple[str, SessionTransition]]) -> TestClient:
    context = SessionContext(verifier=parts.verifier, thresholds=THRESHOLDS)

    def resolve(session_id: str, claimed_subject: str) -> SessionContext | None:
        if session_id == "s-1" and claimed_subject == parts.verifier.template.subject_ref:
            return context
        return None

    def record(session_id: str, transition: SessionTransition, _: SessionContext) -> None:
        recorded.append((session_id, transition))

    return TestClient(create_app(resolve, record))


def test_stream_scores_windows_and_records_the_revocation(
    parts: SessionParts, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_logic_double(monkeypatch, revoke_after=2)
    recorded: list[tuple[str, SessionTransition]] = []
    with client(parts, recorded).websocket_connect("/stream/s-1") as socket:
        socket.send_json(start_message(parts))
        socket.send_bytes(encode_frame(0, parts.probe.data[:, : round(10 * SFREQ)]))
        messages = []
        while not any(m["type"] == "transition" for m in messages):
            messages.append(socket.receive_json())
    windows = [m for m in messages if m["type"] == "window"]
    assert len(windows) == 2
    assert all(isinstance(m["score"], float) for m in windows)
    assert [(sid, t.to_state) for sid, t in recorded] == [("s-1", "revoked")]


def test_unknown_session_is_refused(parts: SessionParts, monkeypatch: pytest.MonkeyPatch) -> None:
    install_logic_double(monkeypatch)
    with client(parts, []).websocket_connect("/stream/unknown") as socket:
        socket.send_json(start_message(parts))
        assert socket.receive_json()["code"] == "bad_start"


def test_stop_closes_the_session(parts: SessionParts, monkeypatch: pytest.MonkeyPatch) -> None:
    install_logic_double(monkeypatch)
    recorded: list[tuple[str, SessionTransition]] = []
    with client(parts, recorded).websocket_connect("/stream/s-1") as socket:
        socket.send_json(start_message(parts))
        socket.send_bytes(encode_frame(0, np.zeros((8, 16))))
        assert socket.receive_json()["type"] == "status"
        socket.send_text('{"type": "stop"}')
        assert socket.receive_json()["to"] == "closed"
    assert [t.to_state for _, t in recorded] == ["closed"]
