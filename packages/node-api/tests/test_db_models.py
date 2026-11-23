from datetime import datetime, timezone

import pytest
from fedbiomed.common.db import get_database
from fedbiomed_node_api.db_models import User, UserRequest
from sqlalchemy import select

_USER = {
    "user_email": "a@b.c",
    "password_hash": "hash",
    "user_name": "n",
    "user_surname": "s",
    "user_role": 2,
    "creation_date": datetime(2026, 1, 2, tzinfo=timezone.utc),
}


@pytest.mark.parametrize(
    "model, data",
    [
        (User, {"user_id": "u", **_USER}),
        (UserRequest, {"request_id": "r", "request_status": "NEW", **_USER}),
    ],
)
def test_gui_tables_round_trip(tmp_path, model, data):
    database = get_database(str(tmp_path / "gui.sqlite"))
    database.create_tables([model])
    with database.transaction() as session:
        session.add(model.from_dict(data))

    with database.transaction() as session:
        (saved,) = session.scalars(select(model)).all()
        assert saved.to_dict() == data
