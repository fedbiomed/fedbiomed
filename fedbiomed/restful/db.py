import uuid
from datetime import datetime
from typing import Dict

from fedbiomed.common.constants import UserRoleType
from fedbiomed.node.dataset_manager._db_tables import (
    UserRequestTable,
    UserTable,
)

from .config import config
from .utils import set_password_hash

user_table = UserTable(config["NODE_DB_PATH"])
user_requests_table = UserRequestTable(config["NODE_DB_PATH"])


def add_default_admin_user(admin_credential: Dict[str, str]) -> None:
    """Adds the default admin user if no admin exists in the node database."""
    email, password = admin_credential["email"], admin_credential["password"]

    try:
        admin = user_table.get_by_role(UserRoleType.ADMIN)
        if not admin:
            print("No admin found, creating default one")
            user_table.insert(
                {
                    "user_email": email,
                    "password_hash": set_password_hash(password),
                    "user_name": "System",
                    "user_surname": "Admin",
                    "user_role": UserRoleType.ADMIN,
                    "creation_date": datetime.utcnow().ctime(),
                    "user_id": "user_" + str(uuid.uuid4()),
                }
            )
    except Exception as e:
        print(f"Error, unable to query in database for admin accounts {e}... resuming")


add_default_admin_user(config["DEFAULT_ADMIN_CREDENTIAL"])
# remove default account credential of env variables
# for security reasons
del config["DEFAULT_ADMIN_CREDENTIAL"]
