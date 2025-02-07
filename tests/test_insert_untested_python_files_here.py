#
# pytest and cobertura results do not show real test coverage
# but only coverage figures of imported files
#
# by including here all .py files of fedbiomed, we force pytest
# to do the right test coverage calculations
#
# TODO: this file may be automatically crafted on ci plateform
# TODO: this file should only contain files not included in proper test_*.py
#       (this is done by hand right now)
#
# find fedbiomed -name '*.py'  | sort | sed -e 's:.py$::' | sed -e 's:/:.:g'
#
import fedbiomed.__init__
import fedbiomed.common.__init__
import fedbiomed.common.certificate_manager
import fedbiomed.common.cli
import fedbiomed.common.constants
import fedbiomed.common.data.__init__
import fedbiomed.common.data._data_loading_plan
import fedbiomed.common.data._data_manager
import fedbiomed.common.data._flamby_dataset
import fedbiomed.common.data._medical_datasets
import fedbiomed.common.data._sklearn_data_manager
import fedbiomed.common.data._tabular_dataset
import fedbiomed.common.data._torch_data_manager
import fedbiomed.common.exceptions
import fedbiomed.common.logger
import fedbiomed.common.message
import fedbiomed.common.metrics
import fedbiomed.common.models.__init__
import fedbiomed.common.models._model
import fedbiomed.common.models._sklearn
import fedbiomed.common.models._torch
import fedbiomed.common.optimizers.optimizer
import fedbiomed.common.optimizers.generic_optimizers
import fedbiomed.common.privacy.__init__
import fedbiomed.common.privacy._dp_controller
import fedbiomed.common.secagg.__init__
import fedbiomed.common.secagg._secagg_crypter
import fedbiomed.common.secagg_manager
import fedbiomed.common.serializer
import fedbiomed.common.singleton
import fedbiomed.common.synchro
import fedbiomed.common.tasks_queue
import fedbiomed.common.training_args
import fedbiomed.common.training_plans.__init__
import fedbiomed.common.training_plans._base_training_plan
import fedbiomed.common.training_plans._sklearn_models
import fedbiomed.common.training_plans._sklearn_models_future
import fedbiomed.common.training_plans._sklearn_training_plan
import fedbiomed.common.training_plans._torchnn
import fedbiomed.common.training_plans._training_iterations
import fedbiomed.common.utils.__init__
import fedbiomed.common.utils._config_utils
import fedbiomed.common.utils._secagg_utils
import fedbiomed.common.utils._utils
import fedbiomed.common.utils._versions
import fedbiomed.common.validator
import fedbiomed.node.__init__
import fedbiomed.node.cli
import fedbiomed.node.cli_utils.__init__
import fedbiomed.node.cli_utils._database
import fedbiomed.node.cli_utils._io
import fedbiomed.node.cli_utils._medical_folder_dataset
import fedbiomed.node.cli_utils._training_plan_management
import fedbiomed.node.dataset_manager
import fedbiomed.node.history_monitor
import fedbiomed.node.training_plan_security_manager
import fedbiomed.node.node
import fedbiomed.node.requests.__init__
import fedbiomed.node.requests._n2n_controller
import fedbiomed.node.requests._n2n_router
import fedbiomed.node.requests._overlay
import fedbiomed.node.round
import fedbiomed.node.secagg
import fedbiomed.node.secagg_manager
import fedbiomed.researcher.__init__
import fedbiomed.researcher.aggregators.__init__
import fedbiomed.researcher.aggregators.aggregator
import fedbiomed.researcher.aggregators.fedavg
import fedbiomed.researcher.aggregators.functional
import fedbiomed.researcher.aggregators.scaffold
import fedbiomed.researcher.cli
import fedbiomed.researcher.datasets
import fedbiomed.researcher.experiment
import fedbiomed.researcher.federated_workflows.__init__
import fedbiomed.researcher.federated_workflows.jobs.__init__
import fedbiomed.researcher.filetools
import fedbiomed.researcher.monitor
import fedbiomed.researcher.requests
import fedbiomed.researcher.secagg.__init__
import fedbiomed.researcher.secagg._secagg_context
import fedbiomed.researcher.secagg._secure_aggregation
import fedbiomed.researcher.strategies.__init__
import fedbiomed.researcher.strategies.default_strategy
import fedbiomed.researcher.strategies.strategy
