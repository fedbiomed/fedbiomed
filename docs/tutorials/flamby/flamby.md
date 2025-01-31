# FLamby integration in Fed-BioMed general concepts

Fed-BioMed supports easy integration with [Owkin's FLamby](https://github.com/owkin/FLamby).
FLamby is a benchmark and dataset suite for cross-silo federated learning with natural partitioning,
focused on healthcare applications. FLamby may be used as either a dataset suite or as a fully-fledged
benchmark to compare the performance of ML algorithms against a set of standardized approaches and data.

Fed-BioMed integration with FLamby is only supported with the PyTorch framework. Hence, to use FLamby you must
declare a `TorchTrainingPlan` for your experiment. Fed-BioMed provides a `FlambyDataset` class that, together
with a correctly configured `DataLoadingPlan`, takes care of all the boilerplate necessary for loading a FLamby
dataset in a federated experiment.

!!! abstract "Summary"
    To use FLamby in your Fed-BioMed experiment, follow these simple rules:

    - use a `TorchTrainingPlan`
    - create a `FlambyDataset` in your `training_data` function
    - Make sure to properly configure a `DataLoadingPlan` when loading the data to the node

## Installing FLamby and downloading the datasets

If you wish to use Flamby, you need to install the package with

```bash
pip install  git+https://github.com/owkin/FLamby@main
```

Additionally, you will need to manually:

- install any dependencies required by the FLamby datasets that you wish to use.
- download those datasets.

To install the dependencies:

* check in FLamby's [setup.py](https://github.com/owkin/FLamby/blob/main/setup.py) the dependencies `<PACKAGES>` for the dataset you wish to use. For example, dataset `tcga` needs `lifelines`.
* install the dependencies by executing on the researcher (where `${FEDBIOMED_DIR}` is Fed-BioMed's base directory)
```bash
# use the python environment for [development](../docs/developer/development-environment.md)
pip install <PACKAGES>
```
* install dependencies by executing on each node
```bash
# use the python environment for [development](../docs/developer/development-environment.md)
pip install <PACKAGES>
```

To download the dataset named `<DATASET>` (eg `fed_ixi` for IXI):

* download the dataset by executing on each node
    - 1. if you are using a **conda virtual environment**:
    ```bash
    # use the python environment for [development](../docs/developer/development-environment.md)
    python $(find $CONDA_PREFIX -path */<DATASET>/dataset_creation_scripts/download.py) -o ${FEDBIOMED_DIR}/data
    ```

    - 2. if you are using a **venv virtual environment**:
    ```bash
    # use the python environment for [development](../docs/developer/development-environment.md)
    python $(find $VIRTUAL_ENV -path */<DATASET>/dataset_creation_scripts/download.py) -o ${FEDBIOMED_DIR}/data
    ```

## Defining the Training Plan

In Fed-BioMed, researchers create a [training plan](../../user-guide/researcher/training-plan.md) 
to define various aspects of their federated ML experiment, such as the model, the data, the optimizer, and others.
To leverage FLamby functionalities within your Fed-BioMed experiment, you will be required to create a 
custom training plan inheriting from `fedbiomed.common.data.TorchTrainingPlan`. 

For details on the meaning of the different functions in a training plan, and how to implement them correctly, 
please follow the [TrainingPlan](../../user-guide/researcher/training-plan.md) user guide. 
Since FLamby is highly compatible with PyTorch, you may use the models and optimizers provided by FLamby in your 
Fed-BioMed experiment seamlessly. See the code below for an example:

```python
from fedbiomed.common.data import TorchTrainingPlan
from flamby.datasets.fed_ixi import Baseline, BaselineLoss, Optimizer

class MyTrainingPlan(TorchTrainingPlan):
    def init_model(self, model_args):
        return Baseline()

    def init_optimizer(self, optimizer_args):
        return Optimizer()

    def init_dependencies(self):
        return ["from flamby.datasets.fed_ixi import Baseline, BaselineLoss, Optimizer"]

    def training_step(self, data, target):
        output = self.model().forward(data)
        return BaselineLoss(output, target)

    def training_data(self):
        # See explanation below
        pass
```

Obviously, you may also plug different definitions for the model, optimizer, and loss function, provided that you
respect the conditions and guidelines for `TorchTrainingPlan`.

## Implementing the `training_data` function

Fed-BioMed provides a `FlambyDataset` class that enables simple integration with FLamby datasets. This class requires 
an associated `DataLoadingPlan` to be properly configured in order to work correctly on the node side. If you follow
the data adding process through either the CLI or the GUI, the configuration of the `DataLoadingPlan` will be done
automatically for you. 

To use Flamby, you need to create a FLamby dataset in your `training_data` function following the example below:
```python
from fedbiomed.common.training_plans import TorchTrainingPlan
from fedbiomed.common.data import FlambyDataset, DataManager


class MyTrainingPlan(TorchTrainingPlan):
    def init_dependencies(self):
        return ["from fedbiomed.common.data import FlambyDataset, DataManager"]

    def training_data(self):
        dataset = FlambyDataset()
        loader_arguments = {'shuffle': True}
        return DataManager(dataset, **loader_arguments)

    # ... Implement the other functions as needed ...
```

## Data transformations

Functional data transformations can be specified in the `training_data` function, similarly to the common 
`TorchTrainingPlan` pattern. However, for FLamby you are required to use the special function `init_transform` of
`FlambyDataset`, as per the example below.

```python
from fedbiomed.common.training_plans import TorchTrainingPlan
from monai.transforms import Compose, Resize, NormalizeIntensity
from fedbiomed.common.data import FlambyDataset, DataManager

class MyTrainingPlan(TorchTrainingPlan):
    def init_dependencies(self):
        return ["from fedbiomed.common.data import FlambyDataset, DataManager",
                "from monai.transforms import Compose, Resize, NormalizeIntensity",
                ]

    def training_data(self):
        dataset = FlambyDataset()
        
        myComposedTransform = Compose([Resize((48,60,48)), NormalizeIntensity()])
        dataset.init_transform(myComposedTransform)

        train_kwargs = {'shuffle': True}
        return DataManager(dataset, **train_kwargs)

    # ... Implement the other functions as needed ...
```

!!! info "Tranforms must always be of `Compose` type"
    Transforms added to a `FlambyDataset` must always be either of type `torch.transforms.Compose` or
    `monai.transforms.Compose`

Do not forget to always add your transform as dependencies in the `init_dependencies` function!
