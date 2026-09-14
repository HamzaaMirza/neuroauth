"""Shared fixtures.

Every fixture here is synthetic. The pipeline tests must run with no eegmmidb
download present, which is only possible because MNE is confined to neuroauth.dsp.io
and everything downstream takes plain arrays. Tests that do need the dataset are
marked @pytest.mark.slow and live in test_io.py.

Signal builders live in tests/synthetic.py so tests can import their constants.
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from neuroauth.dsp.types import Recording
from tests import synthetic


@pytest.fixture
def synthetic_recording() -> Recording:
    """A 60 s, 64-channel recording with a known spectral composition.

    Alpha (10 Hz, 20 uV), beta (20 Hz, 5 uV), and 60 Hz mains (10 uV) over a 2 uV
    noise floor, so band-power and filter assertions can check where power landed.
    """
    return Recording(
        data=synthetic.recording_data(),
        sfreq=synthetic.SFREQ,
        ch_names=synthetic.CH_NAMES,
        subject_id=1,
        run=1,
        condition="eyes_open",
    )


@pytest.fixture
def degraded_windows() -> NDArray[np.float64]:
    """Four windows containing a flat channel, a clipping channel, and a NaN channel.

    The quality mask must flag all three, and extract_features must still return a
    finite matrix of the correct shape rather than raising.
    """
    return synthetic.degraded_windows()
