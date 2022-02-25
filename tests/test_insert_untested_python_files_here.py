#
# nosetests and cobertura results do not show real test coverage
# but only coverage figures off imported files
#
# by including here all .py files of fedbiomed, we force nosetests
# to do the right test coverage calculations
#
# TODO: this file may be automatically crafted on ci plateform
# TODO: this file should only contain files not included in proper test_*.py
#       (this is done by hand right now)
#
#import fedbiomed.common.constants
#import fedbiomed.common.environ
#import fedbiomed.common.exceptions
#import fedbiomed.common.fedbiosklearn
#import fedbiomed.common.json
#import fedbiomed.common.logger
#import fedbiomed.common.message
#import fedbiomed.common.messaging
#import fedbiomed.common.repository
#import fedbiomed.common.singleton
#import fedbiomed.common.tasks_queue
#import fedbiomed.common.torchnn
import fedbiomed.node.cli
#import fedbiomed.node.data_manager
import fedbiomed.node.environ
#import fedbiomed.node.history_monitor
#import fedbiomed.node.model_manager
import fedbiomed.node.node
import fedbiomed.node.round
#import fedbiomed.researcher.aggregators.aggregator
#import fedbiomed.researcher.aggregators.fedavg
import fedbiomed.researcher.aggregators.functional
import fedbiomed.researcher.cli
#import fedbiomed.researcher.datasets
import fedbiomed.researcher.environ
#import fedbiomed.researcher.experiment
#import fedbiomed.researcher.filetools
#import fedbiomed.researcher.job
#import fedbiomed.researcher.monitor
#import fedbiomed.researcher.requests
#import fedbiomed.researcher.responses
import fedbiomed.researcher.strategies.default_strategy
import fedbiomed.researcher.strategies.strategy
