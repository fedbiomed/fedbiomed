"""Node-side fedbiomed tools and CLI."""

# environ is a dependency to all other submodules
from . import environ

# submodules that only depend on environ
from . import dataset_manager
from . import training_plan_security_manager
from . import secagg

# submodules that depend on the other submodules
from . import round  # depends on training_plan_security_manager
from . import node  # depends on all previous submodules

# CLI tools, that depend on the other submodules
from . import cli_utils  # dataset_manager, environ, training_plan_security_manager
from . import cli  # cli_utils, environ, node
