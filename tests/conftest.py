"""Shared fixtures.

Every fixture here is synthetic. The pipeline tests must run with no eegmmidb
download present, which is only possible because MNE is confined to neuroauth.dsp.io
and everything downstream takes plain arrays. Tests that do need the dataset are
marked @pytest.mark.slow and live in test_io.py.
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from neuroauth.dsp.types import Recording

SFREQ = 160.0
N_CHANNELS = 64
DURATION_S = 60.0


@pytest.fixture
def synthetic_recording() -> Recording:
    """A 60 s, 64-channel recording with a known spectral composition.

    Built from summed sinusoids at known frequencies plus a fixed-seed noise floor,
    so band-power assertions can check the right band actually got the power.
    """
    raise NotImplementedError("TODO(phase-1): synthetic recording fixture")


@pytest.fixture
def sine_at() -> object:
    """Factory returning a single-channel sinusoid at a requested frequency.

    Used to assert filter behaviour directly: a 60 Hz tone must be attenuated by the
    notch, a 25 Hz tone must pass the bandpass essentially untouched.
    """
    raise NotImplementedError("TODO(phase-1): sinusoid factory fixture")


@pytest.fixture
def degraded_windows() -> NDArray[np.float64]:
    """Windows containing a flat channel, a clipping channel, and a NaN channel.

    The quality mask must flag all three, and extract_features must still return a
    finite matrix of the correct shape rather than raising.
    """
    raise NotImplementedError("TODO(phase-1): degraded windows fixture")
