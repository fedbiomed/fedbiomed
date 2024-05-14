"""
Sklearn training plans
"""

from fedbiomed.common.training_plans import FedSGDClassifier, FedPerceptron, FedSGDRegressor
from fedbiomed.common.data import DataManager


from fedbiomed.common.optimizers import Optimizer
from fedbiomed.common.optimizers.declearn import AdamModule, FedProxRegularizer, ScaffoldClientModule

import numpy as np
from sklearn.metrics import hinge_loss



class PerceptronTraining(FedPerceptron):
    def training_data(self):
        NUMBER_COLS = 20
        dataset = pd.read_csv(self.dataset_path,header=None,delimiter=',')
        X = dataset.iloc[:,0:NUMBER_COLS].values
        y = dataset.iloc[:,NUMBER_COLS]
        return DataManager(dataset=X,target=y.values, shuffle=True)


class SkLearnClassifierTrainingPlanCustomTesting(FedPerceptron):
    def init_dependencies(self):
        return ["from torchvision import datasets, transforms",
                "from torch.utils.data import DataLoader",
                "from sklearn.metrics import hinge_loss"]

    def compute_accuracy_for_specific_digit(self, data, target, digit: int):
        idx_data_equal_to_digit = (target.squeeze() == digit)

        predicted = self.model().predict(data[idx_data_equal_to_digit])
        well_predicted_label = np.sum(predicted == digit) / np.sum(idx_data_equal_to_digit)
        return well_predicted_label

    def training_data(self):
        # Custom torch Dataloader for MNIST data
        transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
        dataset = datasets.MNIST(self.dataset_path, train=True, download=False, transform=transform)

        train_kwargs = { 'shuffle': True}  # number of data passed to classifier
        X_train = dataset.data.numpy()
        X_train = X_train.reshape(-1, 28*28)
        Y_train = dataset.targets.numpy()

        return DataManager(dataset=X_train, target=Y_train)

    def testing_step(self, data, target):
        # hinge loss
        distance_from_hyperplan = self.model().decision_function(data)
        loss = hinge_loss(target, distance_from_hyperplan)

        # get the accuracy only on images representing digit 1
        well_predicted_label_1 = self.compute_accuracy_for_specific_digit(data, target, 1)

        # Returning results as dict
        return {'Hinge Loss': loss, 'Well Predicted Label 1' : well_predicted_label_1}

class SGDRegressorTrainingPlan(FedSGDRegressor):
    def training_data(self):
        dataset = pd.read_csv(self.dataset_path, delimiter=';', header=0)
        regressors_col = ['AGE', 'WholeBrain.bl',
                          'Ventricles.bl', 'Hippocampus.bl', 'MidTemp.bl', 'Entorhinal.bl']
        target_col = ['MMSE.bl']

        # mean and standard deviation for normalizing dataset
        # it has been computed over the whole dataset
        scaling_mean = np.array([72.3, 0.7, 0.0, 0.0, 0.0, 0.0])
        scaling_sd = np.array([7.3e+00, 5.0e-02, 1.1e-02, 1.0e-03, 2.0e-03, 1.0e-03])

        X = (dataset[regressors_col].values-scaling_mean)/scaling_sd
        y = dataset[target_col]
        return DataManager(dataset=X, target=y.values.ravel(), shuffle=True)

class SGDClassifierTrainingPlan(FedSGDClassifier):
    def training_data(self):
        NUMBER_COLS = 20
        dataset = pd.read_csv(self.dataset_path,header=None,delimiter=',')
        X = dataset.iloc[:,0:NUMBER_COLS].values
        y = dataset.iloc[:,NUMBER_COLS]
        return DataManager(dataset=X,target=y.values, shuffle=True)



class SkLearnClassifierTrainingPlanDeclearn(FedPerceptron):
    def init_dependencies(self):
        """Define additional dependencies.

        In this case, we rely on torchvision functions for preprocessing the images.
        """
        return ["from torchvision import datasets, transforms",
                "from fedbiomed.common.optimizers import Optimizer",
                "from fedbiomed.common.optimizers.declearn import AdamModule, FedProxRegularizer",]

    def training_data(self):
        """Prepare data for training.

        This function loads a MNIST dataset from the node's filesystem, applies some
        preprocessing and converts the full dataset to a numpy array.
        Finally, it returns a DataManager created with these numpy arrays.
        """
        transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
        dataset = datasets.MNIST(self.dataset_path, train=True, download=False, transform=transform)

        X_train = dataset.data.numpy()
        X_train = X_train.reshape(-1, 28*28)
        Y_train = dataset.targets.numpy()
        return DataManager(dataset=X_train, target=Y_train,  shuffle=False)

    # Defines and return a declearn optimizer
    def init_optimizer(self, optimizer_args):
        return Optimizer(lr=.1 ,modules=[AdamModule()], regularizers=[FedProxRegularizer()])


class SGDRegressorTrainingPlanDeclearn(FedSGDRegressor):
    # Declares and return dependencies
    def init_dependencies(self):
        deps = ["from torchvision import datasets, transforms",
                "from declearn.optimizer import Optimizer",
                "from fedbiomed.common.optimizers.declearn import AdamModule",
                "from fedbiomed.common.optimizers.declearn import FedProxRegularizer"]
        return deps

    def training_data(self):
        dataset = pd.read_csv(self.dataset_path, delimiter='[;,]')
        regressors_col = ['AGE', 'WholeBrain.bl',
                          'Ventricles.bl', 'Hippocampus.bl', 'MidTemp.bl', 'Entorhinal.bl']
        target_col = ['MMSE.bl']

        # mean and standard deviation for normalizing dataset
        # it has been computed over the whole dataset
        scaling_mean = np.array([72.3, 0.7, 0.0, 0.0, 0.0, 0.0])
        scaling_sd = np.array([7.3e+00, 5.0e-02, 1.1e-02, 1.0e-03, 2.0e-03, 1.0e-03])

        X = (dataset[regressors_col].values-scaling_mean)/scaling_sd
        y = dataset[target_col]
        return DataManager(dataset=X, target=y.values.ravel(),  shuffle=True)

    # Defines and return a declearn optimizer
    def init_optimizer(self, optimizer_args):
        return Optimizer(lrate=.1 ,modules=[AdamModule()], regularizers=[FedProxRegularizer()])


class SGDRegressorTrainingPlanDeclearnScaffold(FedSGDRegressor):
    # Declares and return dependencies
    def init_dependencies(self):
        deps = ["from torchvision import datasets, transforms",
                "from declearn.optimizer import Optimizer",
                "from fedbiomed.common.optimizers.declearn import AdamModule",
                "from fedbiomed.common.optimizers.declearn import ScaffoldClientModule "]
        return deps

    def training_data(self):
        dataset = pd.read_csv(self.dataset_path, delimiter='[;,]')
        regressors_col = ['AGE', 'WholeBrain.bl',
                          'Ventricles.bl', 'Hippocampus.bl', 'MidTemp.bl', 'Entorhinal.bl']
        target_col = ['MMSE.bl']

        # mean and standard deviation for normalizing dataset
        # it has been computed over the whole dataset
        scaling_mean = np.array([72.3, 0.7, 0.0, 0.0, 0.0, 0.0])
        scaling_sd = np.array([7.3e+00, 5.0e-02, 1.1e-02, 1.0e-03, 2.0e-03, 1.0e-03])

        X = (dataset[regressors_col].values-scaling_mean)/scaling_sd
        y = dataset[target_col]
        return DataManager(dataset=X, target=y.values.ravel(),  shuffle=True)

    # Defines and return a declearn optimizer
    def init_optimizer(self, optimizer_args):

        return Optimizer(lrate=optimizer_args['lr'], modules=[ScaffoldClientModule()])
