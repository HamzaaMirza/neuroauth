"""Score live windows against a claimed identity's protected template.

The inference path. score_windows never raises: a window that cannot be scored becomes a
flagged result, and update_session decides what it is worth.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from neuroauth.config import StreamingConfig
from neuroauth.dsp.types import FeatureMatrix
from neuroauth.templates.enrollment import ProtectedTemplate
from neuroauth.verification.embedding import EmbeddingModel


class TemplateMismatchError(ValueError):
    """The template was made under a different model, transform, or feature configuration.

    Raised at session start, which is not the inference path. Scoring against a template
    from a different feature space would produce confident nonsense rather than an error.
    """


@dataclass(frozen=True, repr=False)
class Verifier:
    """Everything needed to score windows for one claimed identity. In memory, per session.

    repr=False because the projection is derived from key material.

    Attributes:
        model: The embedding the template is bound to.
        template: The claimed identity's current template.
        projection: projection_matrix of the claimed identity's key. The key belongs to the
            claimed identity, never to whoever is producing the signal.
        streaming: Settings the features must be produced with.
    """

    model: EmbeddingModel
    template: ProtectedTemplate
    projection: NDArray[np.float64]
    streaming: StreamingConfig


def load_verifier(
    template: ProtectedTemplate,
    model: EmbeddingModel,
    master_secret: bytes,
    streaming: StreamingConfig,
) -> Verifier:
    """Build a verifier for a claimed identity at session start.

    Raises:
        TemplateMismatchError: If template.model_version, transform_version, or
            streaming_fingerprint differ from the model, cancelable.TRANSFORM_VERSION, or
            streaming.fingerprint().
    """
    raise NotImplementedError("TODO(phase-2): derive key, build projection, check versions")


@dataclass(frozen=True)
class WindowScore:
    """One window's verification result. Holds a score and flags only, so it can be sent and logged.

    Attributes:
        onset_s: Window start, in session time.
        decision_time_s: When the result became available: onset + window_s +
            right_margin_s. Time-to-detect is measured on this clock.
        score: Hamming similarity in [0, 1], or None if the window could not be scored.
        quality_ok: QualityReport.window_ok. Not-ok windows are still scored (D-015).
        flags: Quality flags, plus "feature_contract_mismatch" when scoring was impossible.
    """

    onset_s: float
    decision_time_s: float
    score: float | None
    quality_ok: bool
    flags: tuple[str, ...]


def score_windows(verifier: Verifier, features: FeatureMatrix) -> tuple[WindowScore, ...]:
    """Score every row of a FeatureMatrix against the verifier's template.

    embed, then protect under the claimed identity's projection, then hamming_similarity to
    the template bits. Never raises. A feature_names mismatch, which load_verifier should
    already have made impossible, yields score None with a flag on every row.

    Args:
        verifier: From load_verifier.
        features: Emitted by push_samples, zero or more rows.

    Returns:
        One WindowScore per row, in row order.
    """
    raise NotImplementedError("TODO(phase-2): embed -> protect -> similarity, never raises")
