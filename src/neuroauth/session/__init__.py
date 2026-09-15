"""Continuous sessions: stream in, window scores, state transitions out.

Transport-free. The FastAPI WebSocket endpoint only moves bytes into and out of
runtime.handle_frame, so every session behaviour is testable without a server.
"""
