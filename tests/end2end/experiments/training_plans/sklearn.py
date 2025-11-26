"""
Sklearn training plans
"""

import numpy as np
import pandas as pd
from sklearn.metrics import hinge_loss
from sklearn.pipeline import FunctionTransformer, Pipeline
from torchvision import transforms

from fedbiomed.common.datamanager import DataManager
from fedbiomed.common.dataset import (
    CustomDataset,
    MedNistDataset,
    MnistDataset,
    TabularDataset,
)
from fedbiomed.common.optimizers import Optimizer
from fedbiomed.common.optimizers.declearn import (
    AdamModule,
    FedProxRegularizer,
    ScaffoldClientModule,
)
from fedbiomed.common.training_plans._sklearn_models import (
    FedPerceptron,
    FedSGDClassifier,
    FedSGDRegressor,
)


class NativePerceptronTraining(FedPerceptron):
    def training_data(self):
        dataset = pd.read_csv(self.dataset_path, header=None, delimiter=",")
        _, n_cols = dataset.shape
        X = dataset.iloc[:, : n_cols - 1].values
        y = dataset.iloc[:, n_cols - 1].values
        return DataManager(dataset=X, target=y, shuffle=True)


class NativeSGDRegressorTrainingPlan(FedSGDRegressor):
    def training_data(self):
        dataset = pd.read_csv(self.dataset_path, delimiter=";", header=0)
        regressors_col = [
            "AGE",
            "WholeBrain.bl",
            "Ventricles.bl",
            "Hippocampus.bl",
            "MidTemp.bl",
            "Entorhinal.bl",
        ]
        target_col = ["MMSE.bl"]

        # mean and standard deviation for normalizing dataset
        # it has been computed over the whole dataset
        scaling_mean = np.array([72.3, 0.7, 0.0, 0.0, 0.0, 0.0])
        scaling_sd = np.array([7.3e00, 5.0e-02, 1.1e-02, 1.0e-03, 2.0e-03, 1.0e-03])

        X = (dataset[regressors_col].values - scaling_mean) / scaling_sd
        y = dataset[target_col].values
        return DataManager(dataset=X, target=y, shuffle=True)


class NativeSGDClassifierTrainingPlan(FedSGDClassifier):
    def training_data(self):
        dataset = pd.read_csv(self.dataset_path, header=None, delimiter=",")
        _, n_cols = dataset.shape
        X = dataset.iloc[:, : n_cols - 1].values
        y = dataset.iloc[:, n_cols - 1].values
        return DataManager(dataset=X, target=y, shuffle=True)


class PerceptronTraining(FedPerceptron):
    def init_dependencies(self):
        return [
            "from fedbiomed.common.dataset import TabularDataset",
        ]

    def training_data(self):
        NUMBER_COLS = 20
        dataset = TabularDataset(
            input_columns=list(range(0, NUMBER_COLS)), target_columns=NUMBER_COLS
        )
        return DataManager(dataset=dataset, shuffle=True)


class SkLearnClassifierTrainingPlanCustomTesting(FedPerceptron):
    def init_dependencies(self):
        return [
            "from torchvision import transforms",
            "from torch.utils.data import DataLoader",
            "from sklearn.metrics import hinge_loss",
            "from fedbiomed.common.dataset import MnistDataset",
        ]

    def compute_accuracy_for_specific_digit(self, data, target, digit: int):
        # Filter samples where true label equals the specified digit
        digit_mask = target.squeeze() == digit
        # Get predictions for samples with the target digit
        predictions = self.model().predict(data[digit_mask])
        # Calculate accuracy: correct predictions / total instances of the digit
        accuracy = (predictions == digit).sum() / digit_mask.sum()
        return accuracy

    def training_data(self):
        # Custom torch Dataloader for MNIST data
        transform = transforms.Normalize((0.1307,), (0.3081,))
        dataset = MnistDataset(transform=transform)

        train_kwargs = {"shuffle": True}  # number of data passed to classifier

        return DataManager(dataset=dataset, **train_kwargs)

    def testing_step(self, data, target):
        # hinge loss
        distance_from_hyperplan = self.model().decision_function(data)
        loss = hinge_loss(target, distance_from_hyperplan)

        # get the accuracy only on images representing digit 1
        well_predicted_label_1 = self.compute_accuracy_for_specific_digit(
            data, target, 1
        )

        # Returning results as dict
        return {"Hinge Loss": loss, "Well Predicted Label 1": well_predicted_label_1}


class SGDRegressorTrainingPlan(FedSGDRegressor):
    def init_dependencies(self):
        return ["from fedbiomed.common.dataset import TabularDataset"]

    def training_data(self):
        regressors_col = [
            "AGE",
            "WholeBrain.bl",
            "Ventricles.bl",
            "Hippocampus.bl",
            "MidTemp.bl",
            "Entorhinal.bl",
        ]
        target_col = ["MMSE.bl"]

        def scaling_transform(x: np.ndarray) -> np.ndarray:
            # mean and standard deviation for normalizing dataset
            # it has been computed over the whole dataset
            scaling_mean = np.array([72.3, 0.7, 0.0, 0.0, 0.0, 0.0])
            scaling_sd = np.array([7.3e00, 5.0e-02, 1.1e-02, 1.0e-03, 2.0e-03, 1.0e-03])
            return (x - scaling_mean) / scaling_sd

        dataset = TabularDataset(
            input_columns=regressors_col,
            target_columns=target_col,
            transform=scaling_transform,
        )

        return DataManager(dataset=dataset, shuffle=True)


class SGDClassifierTrainingPlan(FedSGDClassifier):
    def init_dependencies(self):
        return ["from fedbiomed.common.dataset import TabularDataset"]

    def training_data(self):
        NUMBER_COLS = 20
        dataset = TabularDataset(
            input_columns=list(range(0, NUMBER_COLS)), target_columns=NUMBER_COLS
        )
        return DataManager(dataset=dataset, shuffle=True)


class SkLearnClassifierTrainingPlanDeclearn(FedPerceptron):
    def init_dependencies(self):
        """Define additional dependencies.

        In this case, we rely on torchvision functions for preprocessing the images.
        """
        return [
            "from sklearn.pipeline import Pipeline",
            "from sklearn.preprocessing import FunctionTransformer",
            "from fedbiomed.common.optimizers import Optimizer",
            "from fedbiomed.common.optimizers.declearn import AdamModule, FedProxRegularizer",
            "from fedbiomed.common.dataset import MnistDataset",
        ]

    def training_data(self):
        """Prepare data for training.

        This function loads a MNIST dataset from the node's filesystem, applies some
        preprocessing and converts the full dataset to a numpy array.
        Finally, it returns a DataManager created with these numpy arrays.
        """
        # Custom torch Dataloader for MNIST data
        pipeline = Pipeline(
            [
                (
                    "norm",
                    FunctionTransformer(
                        lambda im: (np.asarray(im, dtype=np.float64) / 255.0 - 0.1307)
                        / 0.3081,
                        validate=False,
                    ),
                ),
                (
                    "flatten",
                    FunctionTransformer(
                        lambda x: np.ascontiguousarray(x.reshape(-1), dtype=np.float64),
                        validate=False,
                    ),
                ),
            ]
        )

        dataset = MnistDataset(transform=pipeline.transform)

        return DataManager(dataset=dataset, shuffle=False)

    # Defines and return a declearn optimizer
    def init_optimizer(self, optimizer_args):
        return Optimizer(
            lr=0.1, modules=[AdamModule()], regularizers=[FedProxRegularizer()]
        )


class SGDRegressorTrainingPlanDeclearn(FedSGDRegressor):
    # Declares and return dependencies
    def init_dependencies(self):
        deps = [
            "from torchvision import datasets, transforms",
            "from declearn.optimizer import Optimizer",
            "from fedbiomed.common.optimizers.declearn import AdamModule",
            "from fedbiomed.common.optimizers.declearn import FedProxRegularizer",
            "from fedbiomed.common.dataset import TabularDataset",
        ]
        return deps

    def training_data(self):
        regressors_col = [
            "AGE",
            "WholeBrain.bl",
            "Ventricles.bl",
            "Hippocampus.bl",
            "MidTemp.bl",
            "Entorhinal.bl",
        ]
        target_col = ["MMSE.bl"]

        def scaling_transform(x: np.ndarray) -> np.ndarray:
            # mean and standard deviation for normalizing dataset
            # it has been computed over the whole dataset
            scaling_mean = np.array([72.3, 0.7, 0.0, 0.0, 0.0, 0.0])
            scaling_sd = np.array([7.3e00, 5.0e-02, 1.1e-02, 1.0e-03, 2.0e-03, 1.0e-03])
            return (x - scaling_mean) / scaling_sd

        dataset = TabularDataset(
            input_columns=regressors_col,
            target_columns=target_col,
            transform=scaling_transform,
        )

        return DataManager(dataset=dataset, shuffle=True)

    # Defines and return a declearn optimizer
    def init_optimizer(self, optimizer_args):
        return Optimizer(
            lrate=0.1, modules=[AdamModule()], regularizers=[FedProxRegularizer()]
        )


class SGDRegressorTrainingPlanDeclearnScaffold(FedSGDRegressor):
    # Declares and return dependencies
    def init_dependencies(self):
        deps = [
            "from torchvision import datasets, transforms",
            "from declearn.optimizer import Optimizer",
            "from fedbiomed.common.optimizers.declearn import AdamModule",
            "from fedbiomed.common.optimizers.declearn import ScaffoldClientModule ",
            "from fedbiomed.common.dataset import TabularDataset",
        ]
        return deps

    def training_data(self):
        regressors_col = [
            "AGE",
            "WholeBrain.bl",
            "Ventricles.bl",
            "Hippocampus.bl",
            "MidTemp.bl",
            "Entorhinal.bl",
        ]
        target_col = ["MMSE.bl"]

        def scaling_transform(x: np.ndarray) -> np.ndarray:
            # mean and standard deviation for normalizing dataset
            # it has been computed over the whole dataset
            scaling_mean = np.array([72.3, 0.7, 0.0, 0.0, 0.0, 0.0])
            scaling_sd = np.array([7.3e00, 5.0e-02, 1.1e-02, 1.0e-03, 2.0e-03, 1.0e-03])
            return (x - scaling_mean) / scaling_sd

        dataset = TabularDataset(
            input_columns=regressors_col,
            target_columns=target_col,
            transform=scaling_transform,
        )

        return DataManager(dataset=dataset, shuffle=True)

    # Defines and return a declearn optimizer
    def init_optimizer(self, optimizer_args):
        return Optimizer(lrate=optimizer_args["lr"], modules=[ScaffoldClientModule()])


class SkLearnMedNistTrainingPlan(FedSGDClassifier):
    def init_dependencies(self):
        return [
            "import numpy as np",
            "from sklearn.pipeline import Pipeline",
            "from sklearn.preprocessing import FunctionTransformer",
            "from fedbiomed.common.dataset import MedNistDataset",
        ]

    def training_data(self):
        """
        Builds a numpy-based training dataset from MedNIST
        using a sklearn preprocessing pipeline.
        """

        # ---- Sklearn preprocessing pipeline ----
        # MedNIST images: PIL → grayscale → numpy → flatten
        pipeline = Pipeline(
            [
                (
                    "to_numpy",
                    FunctionTransformer(
                        lambda im: np.asarray(im, dtype=np.float64) / 255.0,
                        validate=False,
                    ),
                ),
                (
                    "flatten",
                    FunctionTransformer(
                        lambda x: np.ascontiguousarray(x.reshape(-1), dtype=np.float64),
                        validate=False,
                    ),
                ),
            ]
        )

        dataset = MedNistDataset(transform=pipeline.transform)

        return DataManager(dataset=dataset, shuffle=True)


class SklearnCSVTrainingPlan(FedSGDClassifier):
    def init_dependencies(self):
        return [
            "import os",
            "import numpy as np",
            "import pandas as pd",
            "from fedbiomed.common.dataset import CustomDataset",
        ]

    class CSVClassificationDataset(CustomDataset):
        """
        CustomDataset for CSV classification datasets generated by
        generate_sklearn_classification_dataset().
        """

        def read(self):
            # The dataset path points directly to the CSV file (ex: c1.csv)
            self.csv_path = self.path

            df = pd.read_csv(self.csv_path, header=None)

            # Last column = label
            self.y = df.iloc[:, -1].to_numpy(dtype=np.int64)

            # All previous columns = features
            self.X = df.iloc[:, :-1].to_numpy(dtype=np.float64)

        def get_item(self, index):
            # Return numpy arrays for sklearn
            x = np.asarray(self.X[index], dtype=np.float64)
            y = np.asarray(self.y[index], dtype=np.int64)
            return x, y

        def __len__(self):
            return self.X.shape[0]

    def training_data(self):
        dataset = self.CSVClassificationDataset()
        return DataManager(dataset=dataset, shuffle=True)
