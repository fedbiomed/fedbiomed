
from training_plans.mnist_pytorch_training_plan import MyTrainingPlan
from fedbiomed.researcher.aggregators.fedavg import FedAverage
from fedbiomed.researcher.experiment import Experiment


# Dfine the experiment
model_args = {}

training_args = {
    'loader_args': { 'batch_size': 48, },
    'optimizer_args': {
        "lr" : 1e-3
    },
    'epochs': 1,
    'dry_run': False,
    'batch_maxnum': 100 # Fast pass for development : only use ( batch_maxnum * batch_size ) samples
}

from fedbiomed.researcher.experiment import Experiment
from fedbiomed.researcher.aggregators.fedavg import FedAverage

tags =  ['#MNIST', '#dataset']
rounds = 2

exp = Experiment(tags=tags,
                 model_args=model_args,
                 training_plan_class=MyTrainingPlan,
                 training_args=training_args,
                 round_limit=rounds,
                 aggregator=FedAverage(),
                 node_selection_strategy=None)


exp.run()
