#
# nosetests and cobertura results do not show real test coverage
# but only coverage figures off imported files
#
# by including here all .py files of fedbiomed, we force nosetests
# to do the right test coverage calculations
#
# TODO: this file may be automatically crafted on ci plateform
#
import fedbiomed.common.tasks_queue
import fedbiomed.common.torchnn
import fedbiomed.common.message
import fedbiomed.common.repository
import fedbiomed.common.json
import fedbiomed.common.messaging
import fedbiomed.node.history_monitor
import fedbiomed.node.round
import fedbiomed.node.cli
import fedbiomed.node.node
import fedbiomed.node.data_manager
import fedbiomed.node.environ
import fedbiomed.researcher.responses
import fedbiomed.researcher.aggregators.aggregator
import fedbiomed.researcher.aggregators.functional
import fedbiomed.researcher.aggregators.fedavg
import fedbiomed.researcher.strategies.default_strategy
import fedbiomed.researcher.strategies.strategy
import fedbiomed.researcher.job
import fedbiomed.researcher.datasets
import fedbiomed.researcher.experiment
import fedbiomed.researcher.cli
import fedbiomed.researcher.exceptions
import fedbiomed.researcher.requests
import fedbiomed.researcher.environ
