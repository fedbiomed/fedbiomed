# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Approval upgrades operate on stored code, without implicitly trusting tutorials."""

import hashlib

import pytest
from python_minifier import minify
from tinydb import TinyDB

from fedbiomed.common.constants import TrainingPlanApprovalStatus
from fedbiomed.common.exceptions import FedbiomedTrainingPlanSecurityManagerError
from fedbiomed.common.message import ApprovalRequest
from fedbiomed.node.training_plan_security_manager import TrainingPlanSecurityManager

SOURCE = "class TutorialPlan:\n    def value(self):\n        return 1\n"
OTHER_SOURCE = "class CustomPlan:\n    def value(self):\n        return 2\n"


def _hash(source, algorithm="SHA256"):
    factories = {
        "SHA256": hashlib.sha256,
        "SHA384": hashlib.sha384,
        "SHA512": hashlib.sha512,
        "SHA3_256": hashlib.sha3_256,
        "SHA3_384": hashlib.sha3_384,
        "SHA3_512": hashlib.sha3_512,
        "BLAKE2B": hashlib.blake2s,
        "BLAKE2S": hashlib.blake2s,
    }
    canonical = minify(
        source,
        remove_annotations=False,
        combine_imports=False,
        remove_pass=False,
        hoist_literals=False,
        remove_object_base=True,
        rename_locals=False,
    )
    return factories[algorithm](canonical.encode("utf-8")).hexdigest()


def _record(
    identifier, source=SOURCE, kind="registered", status="Approved", algorithm="SHA256"
):
    return {
        "name": identifier,
        "description": "Synthetic approval fixture",
        "training_plan_id": identifier,
        "training_plan": source,
        "training_plan_type": kind,
        "training_plan_status": status,
        "algorithm": algorithm,
        "hash": _hash(source, algorithm),
        "researcher_id": "researcher_fixture",
        "notes": "Original administrator notes",
        "date_created": "01-01-2025 00:00:00.000000",
        "date_modified": "01-01-2025 00:00:00.000000",
        "date_registered": "01-01-2025 00:00:00.000000",
        "date_last_action": "01-01-2025 00:00:00.000000",
    }


def _seed(path, records):
    with TinyDB(path) as database:
        database.table("TrainingPlans").insert_multiple(records)


def _manager(path):
    return TrainingPlanSecurityManager(
        str(path), "node_fixture", "Fixture node", "SHA256", True
    )


@pytest.mark.parametrize("gui", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
def test_application_approval_requires_authentication_and_survives_restart(
    tmp_path, monkeypatch, gui, legacy
):
    """Both application factories must use the migrated approval manager."""
    from flask_jwt_extended import create_access_token

    from fedbiomed.node.config import NodeComponent
    from fedbiomed_node_api.application import create_app

    if gui:
        from fedbiomed_gui.server.application import create_app

    root = tmp_path / "node"
    config = NodeComponent().initiate(str(root))
    config.set("security", "training_plan_approval", "True")
    config.write()
    monkeypatch.setenv("DATA_PATH", str(root / "data"))
    if legacy:
        _seed(
            root / "etc" / config.get("default", "db"),
            [_record("tutorial", kind="default", algorithm="SHA512")],
        )

    app = create_app(root, {"TESTING": True})
    manager = app.extensions["node_api_services"]["training_plan_manager"]
    assert manager._tp_approval
    if not legacy:
        assert not manager._db.all()
        request = ApprovalRequest(
            researcher_id="researcher_fixture",
            request_id="request_fixture",
            training_plan=SOURCE,
            description="Synthetic tutorial",
        )
        assert manager.reply_training_plan_approval_request(request).success
    exists, record = manager.check_training_plan_status(
        SOURCE, TrainingPlanApprovalStatus.PENDING
    )
    assert exists
    identifier = record["training_plan_id"]

    for action, status in (
        ("approve", TrainingPlanApprovalStatus.APPROVED),
        ("reject", TrainingPlanApprovalStatus.REJECTED),
    ):
        client = app.test_client()
        body = {"training_plan_id": identifier}
        assert client.post(f"/api/training-plan/{action}", json=body).status_code == 401
        assert not manager.check_training_plan_status(SOURCE, status)[0]
        with app.app_context():
            headers = {
                "Authorization": "Bearer " + create_access_token(identity="test")
            }
        response = client.post(
            f"/api/training-plan/{action}", json=body, headers=headers
        )
        assert response.status_code == 200
        assert response.json["success"]
        manager._tinydb.close()
        app = create_app(root, {"TESTING": True})
        manager = app.extensions["node_api_services"]["training_plan_manager"]
        assert manager.check_training_plan_status(SOURCE, status)[0]
        assert manager.get_training_plan_by_id(identifier)["algorithm"] == "SHA256"
    manager._tinydb.close()


@pytest.mark.parametrize("status", ["Approved", "Pending", "Rejected"])
def test_legacy_defaults_require_explicit_review_without_losing_records(
    tmp_path, status
):
    database = tmp_path / "node.json"
    _seed(database, [_record("tutorial", kind="default", status=status)])
    manager = _manager(database)
    record = manager.get_training_plan_by_id("tutorial")
    assert record["training_plan_type"] == "requested"
    assert record["training_plan_status"] == (
        "Rejected" if status == "Rejected" else "Pending"
    )
    assert record["training_plan"] == SOURCE
    assert record["notes"] == "Original administrator notes"
    manager._tinydb.close()
    restarted = _manager(database)
    assert restarted.get_training_plan_by_id("tutorial") == record
    restarted._tinydb.close()


@pytest.mark.parametrize(
    "algorithm",
    [
        "SHA256",
        "SHA384",
        "SHA512",
        "SHA3_256",
        "SHA3_384",
        "SHA3_512",
        "BLAKE2B",
        "BLAKE2S",
    ],
)
@pytest.mark.parametrize("status", ["Approved", "Pending", "Rejected"])
def test_hash_upgrade_preserves_explicit_decisions_and_metadata(
    tmp_path, algorithm, status
):
    database = tmp_path / "node.json"
    original = _record("custom", algorithm=algorithm, status=status)
    _seed(database, [original])
    manager = _manager(database)
    record = manager.get_training_plan_by_id("custom", secure=False)
    assert record == {**original, "hash": _hash(SOURCE), "algorithm": "SHA256"}
    assert manager.check_training_plan_status(
        SOURCE, TrainingPlanApprovalStatus(status)
    )[0]
    manager._tinydb.close()


@pytest.mark.parametrize(
    "defect", ["wrong_hash", "missing_source", "unknown_algorithm"]
)
def test_invalid_legacy_record_never_partially_migrates_other_approvals(
    tmp_path, defect
):
    database = tmp_path / "node.json"
    records = [
        _record("tutorial", kind="default"),
        _record("invalid", OTHER_SOURCE, algorithm="SHA512"),
    ]
    if defect == "wrong_hash":
        records[1]["hash"] = "not-the-approved-source"
    elif defect == "missing_source":
        records[1].pop("training_plan")
    else:
        records[1]["algorithm"] = "unknown"
    _seed(database, records)
    before = database.read_bytes()
    with pytest.raises(FedbiomedTrainingPlanSecurityManagerError):
        _manager(database)
    assert database.read_bytes() == before


def test_duplicate_code_does_not_choose_between_conflicting_approvals(tmp_path):
    database = tmp_path / "node.json"
    _seed(
        database,
        [_record("tutorial", kind="default"), _record("custom", algorithm="SHA512")],
    )
    before = database.read_bytes()
    with pytest.raises(FedbiomedTrainingPlanSecurityManagerError):
        _manager(database)
    assert database.read_bytes() == before


def test_migrated_tutorial_can_be_approved_rejected_and_reopened(tmp_path):
    database = tmp_path / "node.json"
    _seed(database, [_record("tutorial", kind="default")])
    manager = _manager(database)
    assert not manager.check_training_plan_status(
        SOURCE, TrainingPlanApprovalStatus.APPROVED
    )[0]
    manager.approve_training_plan("tutorial", extra_notes="Reviewed locally")
    assert manager.check_training_plan_status(
        SOURCE, TrainingPlanApprovalStatus.APPROVED
    )[0]
    manager.reject_training_plan("tutorial", extra_notes="Approval revoked")
    manager._tinydb.close()
    restarted = _manager(database)
    assert restarted.check_training_plan_status(
        SOURCE, TrainingPlanApprovalStatus.REJECTED
    )[0]
    restarted._tinydb.close()


def test_fresh_tutorial_request_stays_pending_until_admin_approval(tmp_path):
    manager = _manager(tmp_path / "node.json")
    request = ApprovalRequest(
        researcher_id="researcher_fixture",
        request_id="request_fixture",
        training_plan=SOURCE,
        description="Synthetic tutorial",
    )
    assert manager.reply_training_plan_approval_request(request).success
    found, record = manager.check_training_plan_status(
        SOURCE, TrainingPlanApprovalStatus.PENDING
    )
    assert found
    manager.approve_training_plan(record["training_plan_id"])
    assert manager.check_training_plan_status(
        SOURCE, TrainingPlanApprovalStatus.APPROVED
    )[0]
    manager._tinydb.close()


def test_registration_cannot_reintroduce_automatic_default_type(tmp_path):
    manager = _manager(tmp_path / "node.json")
    source = tmp_path / "tutorial.txt"
    source.write_text(SOURCE)
    with pytest.raises(FedbiomedTrainingPlanSecurityManagerError):
        manager.register_training_plan(
            "tutorial", "Synthetic tutorial", str(source), training_plan_type="default"
        )
    assert manager.list_training_plans(verbose=False) == []
    manager._tinydb.close()
