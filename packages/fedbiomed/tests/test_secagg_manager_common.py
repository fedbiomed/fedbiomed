import pytest
from fedbiomed.common.constants import SecaggElementTypes, __secagg_element_version__
from fedbiomed.common.db import get_database
from fedbiomed.common.db_models import SecaggElement
from fedbiomed.common.exceptions import (
    FedbiomedError,
    FedbiomedSecaggError,
    FedbiomedVersionError,
)
from fedbiomed.common.secagg_manager import SecaggDhManager, SecaggServkeyManager

_SERVKEY = (
    SecaggServkeyManager,
    {"server_key": 123, "biprime": 456},
    SecaggElementTypes.SERVER_KEY,
)
_DH = (
    SecaggDhManager,
    {"node-1": b"DH_KEY_1", "node-2": b"DH_KEY_2"},
    SecaggElementTypes.DIFFIE_HELLMAN,
)


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "component.json")


@pytest.mark.parametrize("manager_class, context, element", [_SERVKEY, _DH])
def test_secagg_manager_add_get_remove(db_path, manager_class, context, element):
    manager = manager_class(db_path)
    assert manager.get("s1", "exp") is None

    manager.add("s1", ["r", "n1", "n2"], context, "exp")
    assert manager.get("s1", "exp") == {
        "secagg_id": "s1",
        "secagg_version": str(__secagg_element_version__),
        "secagg_elem": element.value,
        "experiment_id": "exp",
        "parties": ["r", "n1", "n2"],
        "context": context,
    }

    assert manager.remove("s1", "exp") is True
    assert manager.get("s1", "exp") is None
    assert manager.remove("s1", "exp") is False


@pytest.mark.parametrize("manager_class, context, element", [_SERVKEY, _DH])
def test_secagg_manager_refuses_another_experiment(
    db_path, manager_class, context, element
):
    manager = manager_class(db_path)
    manager.add("s1", ["r", "n1"], context, "exp")

    with pytest.raises(FedbiomedSecaggError, match="does not belong to"):
        manager.get("s1", "another_exp")
    with pytest.raises(FedbiomedSecaggError, match="does not belong to"):
        manager.remove("s1", "another_exp")


def test_secagg_manager_refuses_another_type(db_path):
    SecaggServkeyManager(db_path).add("s1", ["r", "n1"], {"server_key": 1}, "exp")

    with pytest.raises(FedbiomedSecaggError, match="was saved as a"):
        SecaggDhManager(db_path).get("s1", "exp")


@pytest.mark.parametrize("manager_class, context, element", [_SERVKEY, _DH])
def test_secagg_manager_refuses_re_adding(db_path, manager_class, context, element):
    manager = manager_class(db_path)
    manager.add("s1", ["r", "n1"], context, "exp")

    with pytest.raises(FedbiomedError, match="FB634.*UNIQUE constraint failed"):
        manager.add("s1", ["other", "parties"], context, "another_exp")


def test_secagg_manager_refuses_incompatible_version(db_path):
    manager = SecaggServkeyManager(db_path)
    with get_database(db_path).transaction() as session:
        session.add(
            SecaggElement(
                secagg_id="s1",
                secagg_version="99.0",
                secagg_elem=SecaggElementTypes.SERVER_KEY.value,
                experiment_id="exp",
                parties=[],
                context={},
            )
        )

    with pytest.raises(FedbiomedVersionError):
        manager.get("s1", "exp")
