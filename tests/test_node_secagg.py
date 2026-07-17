# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import random
from unittest.mock import patch

import pytest

from fedbiomed.common.constants import SecureAggregationSchemes
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.node.secagg._secagg_round import (
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


@pytest.mark.parametrize(
    "args_fixture, manager_fixture, get_ret, crypter, node_id",
    [
        (
            "lom_args",
            "mock_dhmanager",
            {"parties": ["node-1", "node-2"], "context": {"node-2": b"\x02" * 32}},
            "SecaggLomCrypter",
            "node-1",
        ),
        (
            "jls_args",
            "mock_skmanager",
            {
                "parties": ["researcher-1", "node-1", "node-2"],
                "context": {"server_key": 12345, "biprime": 1156},
            },
            "SecaggCrypter",
            "test-node-id",
        ),
    ],
)
def test_secagg_round_encrypt_forwards_target_range(
    request,
    db,
    mock_skmanager,
    mock_dhmanager,
    args_fixture,
    manager_fixture,
    get_ret,
    crypter,
    node_id,
):
    """Both schemes' round.encrypt() forward target_range to the crypter."""
    request.getfixturevalue(manager_fixture).return_value.get.return_value = get_ret
    with patch(f"fedbiomed.node.secagg._secagg_round.{crypter}") as mock_crypter:
        secagg_round = SecaggRound(
            db=db,
            node_id=node_id,
            secagg_active=True,
            force_secagg=True,
            secagg_arguments=request.getfixturevalue(args_fixture),
            experiment_id="exp-id",
        )
        secagg_round.scheme.encrypt(
            params=[1.0, 1.0], current_round=1, weight=20, target_range=999
        )
    assert mock_crypter.return_value.encrypt.call_args.kwargs["target_range"] == 999


@pytest.mark.parametrize(
    "args_fixture, manager_fixture, get_ret, crypter, node_id",
    [
        (
            "lom_args",
            "mock_dhmanager",
            {"parties": ["node-1", "node-2"], "context": {"node-2": b"\x02" * 32}},
            "SecaggLomCrypter",
            "node-1",
        ),
        (
            "jls_args",
            "mock_skmanager",
            {
                "parties": ["researcher-1", "node-1", "node-2"],
                "context": {"server_key": 12345, "biprime": 1156},
            },
            "SecaggCrypter",
            "test-node-id",
        ),
    ],
)
def test_secagg_round_encrypt_clipping_range_override(
    request,
    db,
    mock_skmanager,
    mock_dhmanager,
    args_fixture,
    manager_fixture,
    get_ret,
    crypter,
    node_id,
):
    """clipping_range overrides the request value; None falls back to it (3)."""
    request.getfixturevalue(manager_fixture).return_value.get.return_value = get_ret
    with patch(f"fedbiomed.node.secagg._secagg_round.{crypter}") as mock_crypter:
        secagg_round = SecaggRound(
            db=db,
            node_id=node_id,
            secagg_active=True,
            force_secagg=True,
            secagg_arguments=request.getfixturevalue(args_fixture),
            experiment_id="exp-id",
        )
        # explicit override forwarded to the crypter
        secagg_round.scheme.encrypt(
            params=[1.0, 1.0], current_round=1, clipping_range=999
        )
        assert (
            mock_crypter.return_value.encrypt.call_args.kwargs["clipping_range"] == 999
        )
        # None falls back to the request's secagg_clipping_range (3)
        secagg_round.scheme.encrypt(params=[1.0, 1.0], current_round=1)
        assert mock_crypter.return_value.encrypt.call_args.kwargs["clipping_range"] == 3


def test_secagg_round_no_secagg_plaintext_path(db, mock_skmanager, mock_dhmanager):
    """No secagg_arguments and force_secagg=False → plaintext path, use_secagg=False."""
    secagg_round = SecaggRound(
        db=db,
        node_id="node-1",
        secagg_arguments=None,
        secagg_active=False,
        force_secagg=False,
        experiment_id="exp-id",
    )
    assert secagg_round.use_secagg is False
    assert secagg_round.scheme is None
