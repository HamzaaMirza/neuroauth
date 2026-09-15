"""Decision layer v0 (D-021): trained on cohort score rows only, outputs a log-likelihood ratio."""

import math

import numpy as np
import pytest

from neuroauth.verification.decision import (
    DECISION_INPUTS,
    decision_inputs,
    decision_llr,
    fit_decision_layer,
)
from neuroauth.verification.metrics import ScoreTable

HOLDOUT = (90,)


def table(
    n_genuine: int = 100,
    n_impostor: int = 1000,
    *,
    impostor_source: int = 2,
    domain: str = "protected",
) -> ScoreTable:
    rng = np.random.default_rng(0)
    scores = np.concatenate((rng.normal(0.75, 0.05, n_genuine), rng.normal(0.5, 0.05, n_impostor)))
    source = np.concatenate((np.full(n_genuine, 1), np.full(n_impostor, impostor_source))).astype(
        np.int64
    )
    n_rows = scores.size
    return ScoreTable(
        scores=scores,
        claimed_subject=np.ones(n_rows, dtype=np.int64),
        source_subject=source,
        source_is_holdout=np.isin(source, HOLDOUT),
        fold=np.zeros(n_rows, dtype=np.int64),
        probe_onset_s=np.arange(n_rows, dtype=np.float64),
        probe_window_ok=np.ones(n_rows, dtype=np.bool_),
        template_score_mean=np.full(n_rows, 0.8),
        template_score_std=np.full(n_rows, 0.05),
        domain=domain,  # type: ignore[arg-type]
        split_kind="cross_condition",
    )


def test_holdout_rows_never_train_the_layer() -> None:
    with pytest.raises(AssertionError, match="impostor-holdout"):
        fit_decision_layer(table(impostor_source=90), n_bits=64, representation_versions=("r",))


def test_only_protected_scores_train_the_layer() -> None:
    with pytest.raises(ValueError, match="protected"):
        fit_decision_layer(table(domain="embedding"), n_bits=64, representation_versions=("r",))


def test_output_is_a_log_likelihood_ratio() -> None:
    """The training prior is subtracted: 100 genuine vs 1000 impostor does not bias the LLR."""
    layer = fit_decision_layer(table(), n_bits=64, representation_versions=("r",))
    assert layer.prior_log_odds == pytest.approx(math.log(100 / 1000))
    assert len(layer.coefficients) == len(DECISION_INPUTS)
    llr = decision_llr(
        layer,
        np.array([0.5, 0.625, 0.75]),
        np.full(3, 0.8),
        np.full(3, 0.05),
        np.ones(3, dtype=np.bool_),
    )
    assert llr[0] < 0.0 < llr[2]
    assert llr[0] < llr[1] < llr[2]


def test_a_zero_template_spread_is_floored_at_one_bit() -> None:
    inputs = decision_inputs(
        np.array([0.75]), np.array([0.75 + 1 / 64]), np.array([0.0]), np.array([True]), n_bits=64
    )
    np.testing.assert_allclose(inputs, [[0.75, -1.0, 1.0]])


def test_decision_version_is_deterministic() -> None:
    first = fit_decision_layer(table(), n_bits=64, representation_versions=("r",))
    second = fit_decision_layer(table(), n_bits=64, representation_versions=("r",))
    assert first.decision_version == second.decision_version
    other = fit_decision_layer(table(), n_bits=64, representation_versions=("s",))
    assert other.decision_version != first.decision_version
