# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import inspect
import random
from unittest.mock import patch

import pytest

from fedbiomed.common.constants import SecureAggregationSchemes
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.node.secagg._secagg_round import (
    SecaggFARound,
    SecaggRound,
    _JLSRound,
    _LomRound,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path):
    return str(tmp_path / "test.json")


@pytest.fixture()
def mock_skmanager():
    with patch(
        "fedbiomed.node.secagg._secagg_round.SecaggServkeyManager", autospec=True
    ) as m:
        yield m


@pytest.fixture()
def mock_dhmanager():
    with patch(
        "fedbiomed.node.secagg._secagg_round.SecaggDhManager", autospec=True
    ) as m:
        yield m


@pytest.fixture()
def jls_args():
    return {
        "secagg_scheme": SecureAggregationSchemes.JOYE_LIBERT,
        "secagg_id": "test-secagg-id",
        "secagg_servkey_id": "test-serv-id",
        "secagg_clipping_range": 3,
        "secagg_random": 34,
        "parties": ["researcher-1", "node-1", "node-2"],
    }


@pytest.fixture()
def lom_args():
    return {
        "secagg_scheme": SecureAggregationSchemes.LOM,
        "secagg_dh_id": "test-dh-id",
        "secagg_clipping_range": 3,
        "secagg_random": 0.5,
        "parties": ["node-1", "node-2"],
    }


# ---------------------------------------------------------------------------
# SecaggRound
# ---------------------------------------------------------------------------


def test_secagg_round_force_secagg_without_arguments_raises(
    db, mock_skmanager, mock_dhmanager
):
    with pytest.raises(FedbiomedError):
        SecaggRound(
            db=db,
            node_id="test-node-id",
            force_secagg=True,
            secagg_active=True,
            secagg_arguments={},
            experiment_id="test-id",
        )


def test_secagg_round_secagg_not_active_raises(
    db, mock_skmanager, mock_dhmanager, jls_args
):
    with pytest.raises(FedbiomedError):
        SecaggRound(
            db=db,
            node_id="test-node-id",
            secagg_active=False,
            force_secagg=False,
            secagg_arguments=jls_args,
            experiment_id="test-id",
        )


def test_secagg_round_missing_scheme_raises(
    db, mock_skmanager, mock_dhmanager, jls_args
):
    jls_args.pop("secagg_scheme")
    with pytest.raises(FedbiomedError):
        SecaggRound(
            db=db,
            node_id="test-node-id",
            secagg_active=True,
            force_secagg=True,
            secagg_arguments=jls_args,
            experiment_id="test-id",
        )


def test_secagg_round_bad_scheme_value_raises(
    db, mock_skmanager, mock_dhmanager, jls_args
):
    jls_args["secagg_scheme"] = "oops"
    with pytest.raises(FedbiomedError):
        SecaggRound(
            db=db,
            node_id="test-node-id",
            secagg_active=True,
            force_secagg=True,
            secagg_arguments=jls_args,
            experiment_id="test-id",
        )


def test_secagg_round_jls_instantiation(db, mock_skmanager, mock_dhmanager, jls_args):
    mock_skmanager.return_value.get.return_value = {
        "parties": ["researcher-1", "node-1", "node-2"]
    }
    secagg_round = SecaggRound(
        db=db,
        node_id="test-node-id",
        secagg_active=True,
        force_secagg=True,
        secagg_arguments=jls_args,
        experiment_id="test-id",
    )
    assert isinstance(secagg_round.scheme, _JLSRound)
    assert secagg_round.scheme.secagg_random == 34


def test_secagg_round_jls_parties_mismatch_raises(
    db, mock_skmanager, mock_dhmanager, jls_args
):
    mock_skmanager.return_value.get.return_value = {
        "parties": ["researcher-99", "node-1", "node-2"]
    }
    with pytest.raises(FedbiomedError):
        SecaggRound(
            db=db,
            node_id="test-node-id",
            secagg_active=True,
            force_secagg=True,
            secagg_arguments=jls_args,
            experiment_id="test-id",
        )


def test_secagg_round_jls_context_missing_raises(
    db, mock_skmanager, mock_dhmanager, jls_args
):
    mock_skmanager.return_value.get.return_value = None
    with pytest.raises(FedbiomedError):
        SecaggRound(
            db=db,
            node_id="test-node-id",
            secagg_active=True,
            force_secagg=True,
            secagg_arguments=jls_args,
            experiment_id="test-id",
        )


def test_secagg_round_jls_too_few_parties_raises(
    db, mock_skmanager, mock_dhmanager, jls_args
):
    jls_args["parties"] = ["only-one"]
    with pytest.raises(FedbiomedError):
        SecaggRound(
            db=db,
            node_id="test-node-id",
            secagg_active=True,
            force_secagg=True,
            secagg_arguments=jls_args,
            experiment_id="test-id",
        )


def test_secagg_round_jls_invalid_clipping_range_raises(
    db, mock_skmanager, mock_dhmanager, jls_args
):
    jls_args["secagg_clipping_range"] = "invalid-type"
    with pytest.raises(FedbiomedError):
        SecaggRound(
            db=db,
            node_id="test-node-id",
            secagg_active=True,
            force_secagg=True,
            secagg_arguments=jls_args,
            experiment_id="test-id",
        )


def test_secagg_round_jls_encrypt(db, mock_skmanager, mock_dhmanager, jls_args):
    mock_skmanager.return_value.get.return_value = {
        "parties": ["researcher-1", "node-1", "node-2"],
        "context": {"server_key": 12345, "biprime": 1156},
    }
    secagg = SecaggRound(
        db=db,
        node_id="test-node-id",
        secagg_active=True,
        force_secagg=True,
        secagg_arguments=jls_args,
        experiment_id="test-id",
    )
    result = secagg.scheme.encrypt(params=[1.0, 1.0], current_round=1, weight=20)
    assert isinstance(result, list)
    assert len(result) > 0


def test_secagg_round_lom_instantiation(db, mock_skmanager, mock_dhmanager, lom_args):
    mock_dhmanager.return_value.get.return_value = {"parties": ["node-1", "node-2"]}
    secagg_round = SecaggRound(
        db=db,
        node_id="test-node-id",
        secagg_active=True,
        force_secagg=True,
        secagg_arguments=lom_args,
        experiment_id="test-exp-id",
    )
    assert isinstance(secagg_round.scheme, _LomRound)


def test_secagg_round_lom_context_missing_raises(
    db, mock_skmanager, mock_dhmanager, lom_args
):
    mock_dhmanager.return_value.get.return_value = None
    with pytest.raises(FedbiomedError):
        SecaggRound(
            db=db,
            node_id="test-node-id",
            secagg_active=True,
            force_secagg=True,
            secagg_arguments=lom_args,
            experiment_id="test-exp-id",
        )


def test_secagg_round_lom_parties_mismatch_raises(
    db, mock_skmanager, mock_dhmanager, lom_args
):
    mock_dhmanager.return_value.get.return_value = {"parties": ["no-match"]}
    with pytest.raises(FedbiomedError):
        SecaggRound(
            db=db,
            node_id="test-node-id",
            secagg_active=True,
            force_secagg=True,
            secagg_arguments=lom_args,
            experiment_id="test-exp-id",
        )


def test_secagg_round_lom_encrypt(db, mock_skmanager, mock_dhmanager, lom_args):
    mock_dhmanager.return_value.get.return_value = {
        "parties": ["node-1", "node-2"],
        "context": {"node-2": random.randbytes(32)},
    }
    secagg_round = SecaggRound(
        db=db,
        node_id="node-1",
        secagg_active=True,
        force_secagg=True,
        secagg_arguments=lom_args,
        experiment_id="test-exp-id",
    )
    result = secagg_round.scheme.encrypt(params=[1.0, 1.0], current_round=1, weight=20)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# SecaggFARound
# ---------------------------------------------------------------------------


@pytest.fixture()
def fa_jls_args():
    return {
        "secagg_scheme": SecureAggregationSchemes.JOYE_LIBERT,
        "secagg_servkey_id": "test-serv-id",
        "secagg_clipping_range": 3,
        "secagg_random": 0.5,
        "parties": ["researcher-1", "node-1", "node-2"],
        "fa_round": 1,
    }


@pytest.fixture()
def fa_lom_args():
    return {
        "secagg_scheme": SecureAggregationSchemes.LOM,
        "secagg_dh_id": "test-dh-id",
        "secagg_clipping_range": 3,
        "secagg_random": 0.5,
        "parties": ["node-1", "node-2"],
        "fa_round": 1,
    }


def _make_fa_round(db, args, secagg_active=True, force_secagg=False):
    """Helper: construct SecaggFARound with explicit policy flags."""
    return SecaggFARound(
        db=db,
        node_id="node-1",
        secagg_arguments=args,
        secagg_active=secagg_active,
        force_secagg=force_secagg,
        experiment_id="exp-id",
    )


# --- policy guards (mirror SecaggRound behaviour) ---


def test_fa_round_force_secagg_without_arguments_raises(
    db, mock_skmanager, mock_dhmanager
):
    """force_secagg=True + no secagg_arguments → raises (node mandates secagg)."""
    with pytest.raises(FedbiomedError):
        _make_fa_round(db, args=None, secagg_active=True, force_secagg=True)


def test_fa_round_secagg_not_active_raises(
    db, mock_skmanager, mock_dhmanager, fa_jls_args
):
    """secagg_arguments provided + secagg_active=False → raises."""
    with pytest.raises(FedbiomedError):
        _make_fa_round(db, fa_jls_args, secagg_active=False, force_secagg=False)


def test_fa_round_no_args_no_force_succeeds(db, mock_skmanager, mock_dhmanager):
    """No secagg_arguments and force_secagg=False → plaintext path, use_secagg=False."""
    fa_round = _make_fa_round(db, args=None, secagg_active=False, force_secagg=False)
    assert fa_round.use_secagg is False
    assert fa_round.scheme is None


# --- scheme selection and context validation ---


def test_fa_round_missing_scheme_raises(
    db, mock_skmanager, mock_dhmanager, fa_jls_args
):
    fa_jls_args.pop("secagg_scheme")
    with pytest.raises(FedbiomedError):
        _make_fa_round(db, fa_jls_args)


def test_fa_round_invalid_scheme_raises(
    db, mock_skmanager, mock_dhmanager, fa_jls_args
):
    fa_jls_args["secagg_scheme"] = "NOT_A_SCHEME"
    with pytest.raises(FedbiomedError):
        _make_fa_round(db, fa_jls_args)


def test_fa_round_jls_instantiation(db, mock_skmanager, mock_dhmanager, fa_jls_args):
    mock_skmanager.return_value.get.return_value = {
        "parties": ["researcher-1", "node-1", "node-2"]
    }
    fa_round = _make_fa_round(db, fa_jls_args)
    assert isinstance(fa_round.scheme, _JLSRound)
    assert fa_round.use_secagg is True


def test_fa_round_jls_context_missing_raises(
    db, mock_skmanager, mock_dhmanager, fa_jls_args
):
    mock_skmanager.return_value.get.return_value = None
    with pytest.raises(FedbiomedError):
        _make_fa_round(db, fa_jls_args)


def test_fa_round_jls_parties_mismatch_raises(
    db, mock_skmanager, mock_dhmanager, fa_jls_args
):
    mock_skmanager.return_value.get.return_value = {
        "parties": ["researcher-99", "node-1", "node-2"]
    }
    with pytest.raises(FedbiomedError):
        _make_fa_round(db, fa_jls_args)


def test_fa_round_jls_encrypt(db, mock_skmanager, mock_dhmanager, fa_jls_args):
    mock_skmanager.return_value.get.return_value = {
        "parties": ["researcher-1", "node-1", "node-2"],
        "context": {"server_key": 12345, "biprime": 1156},
    }
    fa_round = _make_fa_round(db, fa_jls_args)
    # JLS returns a single aggregate ciphertext (not per-element).
    result = fa_round.encrypt([1.0, 2.0, 3.0], fa_round=1, weight=1)
    assert isinstance(result, list)
    assert len(result) > 0


def test_fa_round_lom_instantiation(db, mock_skmanager, mock_dhmanager, fa_lom_args):
    mock_dhmanager.return_value.get.return_value = {"parties": ["node-1", "node-2"]}
    fa_round = _make_fa_round(db, fa_lom_args)
    assert isinstance(fa_round.scheme, _LomRound)
    assert fa_round.use_secagg is True


def test_fa_round_lom_context_missing_raises(
    db, mock_skmanager, mock_dhmanager, fa_lom_args
):
    mock_dhmanager.return_value.get.return_value = None
    with pytest.raises(FedbiomedError):
        _make_fa_round(db, fa_lom_args)


def test_fa_round_lom_parties_mismatch_raises(
    db, mock_skmanager, mock_dhmanager, fa_lom_args
):
    mock_dhmanager.return_value.get.return_value = {"parties": ["no-match"]}
    with pytest.raises(FedbiomedError):
        _make_fa_round(db, fa_lom_args)


def test_fa_round_lom_encrypt(db, mock_skmanager, mock_dhmanager, fa_lom_args):
    mock_dhmanager.return_value.get.return_value = {
        "parties": ["node-1", "node-2"],
        "context": {"node-2": random.randbytes(32)},
    }
    fa_round = _make_fa_round(db, fa_lom_args)
    result = fa_round.encrypt([1.0, 2.0, 3.0], fa_round=1, weight=1)
    assert len(result) == 3


def test_fa_round_has_policy_guard_params():
    """SecaggFARound.__init__ must accept secagg_active and force_secagg."""
    sig = inspect.signature(SecaggFARound.__init__)
    assert "secagg_active" in sig.parameters
    assert "force_secagg" in sig.parameters
