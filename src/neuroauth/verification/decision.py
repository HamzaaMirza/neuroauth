"""Decision layer v0: score-level inputs to a calibrated log-likelihood ratio (D-021).

The representation (bounded-context features, embedding, cancelable transform) is frozen at
enrollment. What the correction loop retrains, from Phase 3 on, is this layer. Its inputs are
only what may be persisted about a window, never a feature vector (hard rule 3):

- the Hamming similarity s to the claimed template;
- its z-score against the template's enrollment statistics, (s - mu) / max(sigma, 1/n_bits),
  which lets the layer learn per-template score offsets;
- the window's quality flag.

A single monotone calibration of s would leave the pooled DET curve, and so the EER,
unchanged, and could never pass the Phase 3 gate. That is why the layer takes inputs beyond
the score. Per-bit inputs are impossible: every subject's projection is an independent
random rotation, so bit i means something different for each user.

The logistic output is a log posterior odds under the training set's class balance; the
training log prior odds are subtracted, so the result is a log-likelihood ratio.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression

from neuroauth.verification.metrics import ScoreTable, genuine_mask

DECISION_ALGORITHM: Final = "logistic-v0/similarity,template-z,window-ok"
DECISION_INPUTS: Final = ("similarity", "template_z", "window_ok")
DECISION_C: Final = 1.0
DECISION_MAX_ITER: Final = 1000
"""Hyperparameters fixed before any Phase 2 result (D-021)."""


def decision_inputs(
    similarity: NDArray[np.float64],
    template_score_mean: NDArray[np.float64],
    template_score_std: NDArray[np.float64],
    window_ok: NDArray[np.bool_],
    *,
    n_bits: int,
) -> NDArray[np.float64]:
    """(n, 3) inputs in DECISION_INPUTS order.

    The standard deviation is floored at one bit step, 1 / n_bits, so a template whose
    enrollment windows agreed perfectly does not produce an infinite z-score.
    """
    if n_bits < 1:
        raise ValueError(f"n_bits must be at least 1, got {n_bits}")
    s = np.asarray(similarity, dtype=np.float64)
    sigma = np.maximum(np.asarray(template_score_std, dtype=np.float64), 1.0 / n_bits)
    z = (s - np.asarray(template_score_mean, dtype=np.float64)) / sigma
    return np.column_stack((s, z, np.asarray(window_ok, dtype=np.float64)))


@dataclass(frozen=True)
class DecisionLayer:
    """A fitted decision layer. Holds coefficients over score-level inputs only.

    Attributes:
        coefficients: (3,) in DECISION_INPUTS order.
        intercept: Logistic intercept.
        prior_log_odds: log(n_genuine / n_impostor) of the training rows, subtracted from
            the logit so the output is a log-likelihood ratio.
        n_bits: Template bit count the z-score floor uses.
        representation_versions: Representations the training scores came from.
        n_genuine: Genuine training rows.
        n_impostor: Impostor training rows, all cohort impostors.
        decision_version: "sha256:" plus 16 hex digits over everything above.
    """

    coefficients: NDArray[np.float64]
    intercept: float
    prior_log_odds: float
    n_bits: int
    representation_versions: tuple[str, ...]
    n_genuine: int
    n_impostor: int
    decision_version: str


def fit_decision_layer(
    table: ScoreTable,
    *,
    n_bits: int,
    representation_versions: tuple[str, ...],
) -> DecisionLayer:
    """Fit the layer on protected-domain genuine and cohort-impostor rows.

    Raises:
        AssertionError: If any row's source is an impostor-holdout subject. The holdout is
            for reporting only; the hard-rule-5 posture carries over to score rows.
        ValueError: If the table is not protected-domain, lacks either class, or has
            non-finite template statistics.
    """
    if table.domain != "protected":
        raise ValueError(f"the decision layer trains on protected scores, got {table.domain}")
    if table.source_is_holdout.any():
        raise AssertionError("impostor-holdout rows in decision-layer training data")
    genuine = genuine_mask(table)
    n_genuine, n_impostor = int(genuine.sum()), int((~genuine).sum())
    if n_genuine == 0 or n_impostor == 0:
        raise ValueError("decision-layer training needs genuine and impostor rows")
    x = decision_inputs(
        table.scores,
        table.template_score_mean,
        table.template_score_std,
        table.probe_window_ok,
        n_bits=n_bits,
    )
    if not np.isfinite(x).all():
        raise ValueError("non-finite decision inputs; template statistics are required")

    fitted = LogisticRegression(C=DECISION_C, max_iter=DECISION_MAX_ITER, solver="lbfgs")
    fitted.fit(x, genuine.astype(np.int64))
    coefficients = np.asarray(fitted.coef_, dtype=np.float64).reshape(-1)
    intercept = float(np.asarray(fitted.intercept_).reshape(-1)[0])
    prior_log_odds = math.log(n_genuine / n_impostor)

    header = {
        "algorithm": DECISION_ALGORITHM,
        "c": DECISION_C,
        "intercept": intercept,
        "n_bits": n_bits,
        "n_genuine": n_genuine,
        "n_impostor": n_impostor,
        "prior_log_odds": prior_log_odds,
        "representation_versions": list(representation_versions),
    }
    digest = hashlib.sha256(json.dumps(header, sort_keys=True).encode("utf-8"))
    digest.update(np.ascontiguousarray(coefficients, dtype="<f8").tobytes())
    return DecisionLayer(
        coefficients=coefficients,
        intercept=intercept,
        prior_log_odds=prior_log_odds,
        n_bits=n_bits,
        representation_versions=representation_versions,
        n_genuine=n_genuine,
        n_impostor=n_impostor,
        decision_version=f"sha256:{digest.hexdigest()[:16]}",
    )


def decision_llr(
    layer: DecisionLayer,
    similarity: NDArray[np.float64],
    template_score_mean: NDArray[np.float64],
    template_score_std: NDArray[np.float64],
    window_ok: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Log-likelihood ratio per window. Never raises on content; non-finite inputs give NaN."""
    x = decision_inputs(
        similarity, template_score_mean, template_score_std, window_ok, n_bits=layer.n_bits
    )
    with np.errstate(invalid="ignore", over="ignore"):
        llr: NDArray[np.float64] = x @ layer.coefficients + layer.intercept - layer.prior_log_odds
    return llr
