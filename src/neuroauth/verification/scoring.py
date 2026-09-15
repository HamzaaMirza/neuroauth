"""Score live windows against a claimed identity's protected template.

The inference path. score_windows never raises: a window that cannot be scored becomes a
flagged result, and update_session decides what it is worth.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from neuroauth.config import StreamingConfig
from neuroauth.dsp.types import FeatureMatrix
from neuroauth.templates.cancelable import (
    TRANSFORM_VERSION,
    derive_key,
    hamming_similarity,
    projection_matrix,
    protect,
)
from neuroauth.templates.enrollment import ProtectedTemplate, representation_version
from neuroauth.verification.decision import DecisionLayer, decision_llr
from neuroauth.verification.embedding import EmbeddingModel, embed_values


class TemplateMismatchError(ValueError):
    """The template was made under a different representation, or the decision layer was not
    trained on it.

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
        decision: The decision layer in effect.
        projection: projection_matrix of the claimed identity's key. The key belongs to the
            claimed identity, never to whoever is producing the signal.
        streaming: Settings the features must be produced with.
    """

    model: EmbeddingModel
    template: ProtectedTemplate
    decision: DecisionLayer
    projection: NDArray[np.float64]
    streaming: StreamingConfig


def load_verifier(
    template: ProtectedTemplate,
    model: EmbeddingModel,
    decision: DecisionLayer,
    master_secret: bytes,
    streaming: StreamingConfig,
) -> Verifier:
    """Build a verifier for a claimed identity at session start.

    Raises:
        TemplateMismatchError: If the template's embedding, transform, streaming
            fingerprint, or representation version differ from the model and settings, or
            the decision layer was not trained on that representation.
    """
    fingerprint = streaming.fingerprint()
    problems = []
    if template.embedding_version != model.embedding_version:
        problems.append("embedding version")
    if template.transform_version != TRANSFORM_VERSION:
        problems.append("transform version")
    if template.streaming_fingerprint != fingerprint or model.streaming_fingerprint != fingerprint:
        problems.append("streaming fingerprint")
    if template.representation_version != representation_version(model):
        problems.append("representation version")
    if template.representation_version not in decision.representation_versions:
        problems.append("decision layer representation")
    if problems:
        raise TemplateMismatchError(
            f"template for {template.subject_ref} does not match: {', '.join(problems)}"
        )
    key = derive_key(master_secret, template.subject_ref, template.key_version)
    return Verifier(
        model=model,
        template=template,
        decision=decision,
        projection=projection_matrix(key, model.n_components, model.n_components),
        streaming=streaming,
    )


@dataclass(frozen=True)
class WindowScore:
    """One window's verification result. Holds scores and flags only, so it can be sent and logged.

    Attributes:
        onset_s: Window start, in session time.
        decision_time_s: When the result became available: onset + window_s +
            right_margin_s. Time-to-detect is measured on this clock.
        score: Hamming similarity in [0, 1], or None if the window could not be scored.
        llr: The decision layer's log-likelihood ratio (D-021), or None if unscorable.
        quality_ok: QualityReport.window_ok. Not-ok windows are still scored (D-015).
        flags: Quality flags, plus "feature_contract_mismatch" or "non_finite_embedding"
            when scoring was impossible.
    """

    onset_s: float
    decision_time_s: float
    score: float | None
    llr: float | None
    quality_ok: bool
    flags: tuple[str, ...]


def score_windows(verifier: Verifier, features: FeatureMatrix) -> tuple[WindowScore, ...]:
    """Score every row of a FeatureMatrix against the verifier's template.

    embed, then protect under the claimed identity's projection, then hamming_similarity to
    the template bits, then the decision layer. Never raises. A feature_names or shape
    mismatch, which load_verifier and the stream settings should already have made
    impossible, yields score None with a flag on every row.

    Args:
        verifier: From load_verifier.
        features: Emitted by push_samples, zero or more rows.

    Returns:
        One WindowScore per row, in row order.
    """
    streaming = verifier.streaming
    delay_s = streaming.pipeline.window.window_s + streaming.context.right_margin_s
    onsets = np.asarray(features.onsets_s, dtype=np.float64).reshape(-1)
    window_ok = np.asarray(features.quality.window_ok, dtype=np.bool_).reshape(-1)
    values = np.asarray(features.values)
    n_rows = onsets.size
    usable = (
        features.feature_names == verifier.model.feature_names
        and values.ndim == 2
        and values.shape == (n_rows, len(verifier.model.feature_names))
        and window_ok.size == n_rows
        and len(features.quality.flags) == n_rows
    )

    def result(
        i: int, score: float | None, llr: float | None, extra: tuple[str, ...]
    ) -> WindowScore:
        ok = bool(window_ok[i]) if i < window_ok.size else False
        flags = features.quality.flags[i] if i < len(features.quality.flags) else ()
        return WindowScore(
            onset_s=float(onsets[i]),
            decision_time_s=float(onsets[i]) + delay_s,
            score=score,
            llr=llr,
            quality_ok=ok,
            flags=(*flags, *extra),
        )

    if not usable:
        return tuple(result(i, None, None, ("feature_contract_mismatch",)) for i in range(n_rows))
    if n_rows == 0:
        return ()

    with np.errstate(invalid="ignore", over="ignore"):
        embeddings = embed_values(verifier.model, values.astype(np.float64))
    finite = np.isfinite(embeddings).all(axis=1)
    bits = protect(np.where(finite[:, None], embeddings, 0.0), verifier.projection)
    similarity = hamming_similarity(bits, verifier.template.bits)
    template = verifier.template
    llr = decision_llr(
        verifier.decision,
        similarity,
        np.full(n_rows, template.enrollment_score_mean),
        np.full(n_rows, template.enrollment_score_std),
        window_ok,
    )
    scores: list[WindowScore] = []
    for i in range(n_rows):
        if not finite[i]:
            scores.append(result(i, None, None, ("non_finite_embedding",)))
            continue
        llr_i = float(llr[i]) if np.isfinite(llr[i]) else None
        scores.append(result(i, float(similarity[i]), llr_i, ()))
    return tuple(scores)
