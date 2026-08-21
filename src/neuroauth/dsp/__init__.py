"""Signal processing.

Named `dsp` rather than `signal` so it never shadows the stdlib module that
deployment code will want for handlers.

MNE is confined to `neuroauth.dsp.io`. Every other module here is numpy in, numpy
out, with no I/O and no MNE object, so the same preprocessing, windowing, and
feature code runs unchanged over a live WebSocket stream in Phase 2.
"""
