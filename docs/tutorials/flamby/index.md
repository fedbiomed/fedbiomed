# FLamby Integration in Fed-BioMed: General Concepts

Fed-BioMed supports easy integration with [Owkin's FLamby](https://github.com/owkin/FLamby).
FLamby is a benchmark and dataset suite for cross-silo federated learning with natural partitioning,
focused on healthcare applications. FLamby can be used either as a dataset suite or as a fully-fledged
benchmark to compare the performance of ML algorithms against a set of standardized approaches and data.

Fed-BioMed's integration with FLamby is only supported with the PyTorch framework. Therefore, to use FLamby you must
declare a `TorchTrainingPlan` for your experiment. Fed-BioMed provides a `FlambyDataset` class that, together
with a correctly configured `DataLoadingPlan`, handles all the boilerplate necessary for loading a FLamby
dataset in a federated experiment.

!!! abstract "Summary"
    To use FLamby in your Fed-BioMed experiment, follow these simple rules:

    - Use a `TorchTrainingPlan`
    - Create a `FlambyDataset` custom dataset in your `training_data` function
    - Make sure to properly configure a `DataLoadingPlan` when loading the data to the node

## Installing FLamby and Downloading the Datasets

If you wish to use FLamby, you need to install the package with:

```bash
pip install git+https://github.com/owkin/FLamby@main
```

Additionally, you will need to manually:

- Install any dependencies required by the FLamby datasets that you wish to use
- Download those datasets

To install the dependencies:

* Check FLamby's [setup.py](https://github.com/owkin/FLamby/blob/main/setup.py) for the dependencies `<PACKAGES>` for the dataset you wish to use. For example, the `tcga` dataset requires `lifelines`.

Here are the packages that you may need to install to follow the tutorials under Fed-BioMed documentation that use FLamby: 

```bash
# Use the python environment for development (see docs/developer/development-environment.md)
pip install wget
```

## Understanding FLamby Dataset Structure

Each FLamby dataset follows a consistent structure:

```
flamby/datasets/[dataset_name]/
├── dataset_creation_scripts/
│   ├── download.py          # Main download script
│   └── update_config.py     # Configuration update script
├── README.md               # Dataset documentation
└── other dataset files...
```

The `download.py` script in each dataset's `dataset_creation_scripts` folder is responsible for downloading the actual data.

### Download Scripts

FLamby datasets come with scripts located within the library to download datasets. However, these scripts are not exposed through a CLI utility. Therefore, users should locate the dataset directory and find the download script to download the datasets.

Here is a script that you can use to locate these files: 

```python
import flamby.datasets.fed_heaart_desease

# Select the dataset 
dataset = flamby.datasets.fed_heart_disease

# Find the dataset root file 
dataset_root = Path(dataset.__file__).parent
```

The download scripts are located under the root directory of dataset modules: 

```python
download_script = dataset_root / "dataset_creation_scripts" / "download.py"
update_script = dataset_root / "dataset_creation_scripts" / "update_config.py"
```

Executing the script from Jupyter Notebook and Python shell or script may differ. The `!python` command can be used for executing the script in Jupyter notebook, and `os.system` for python shell or script:   

```python
# Jupyter notebook
!python {download_script} --output-folder <path-to-download-folder>
```

```python
# Python
os.system(f"python {download_script} --output-folder <path-to-download-folder>")
```

After downloading the dataset, FLamby creates a YAML file called `dataset_location.yaml` which is located under the dataset module's root directory. This file keeps the information about where the data is downloaded: 

```python
dataset_yaml = dataset_root / "dataset_creation_scripts" / "dataset_location.yaml"
os.system(f"cat {dataset_yaml}")
```

!!! note "Re-downloading dataset"
    To be able to re-download the dataset, this YAML file has to be removed. 


### Update Script

FLamby datasets also come with update scripts to update the location where the dataset is downloaded:

```python
update_script = dataset_root / "dataset_creation_scripts" / "update_config.py"
os.system(f"python {update_script} --new-path </new/path/towards/dataset>")
```

### Adding Dataset Nodes Using CLI

FLamby datasets have to be deployed on the nodes to be used in Fed-BioMed federated training. Fed-BioMed provides an abstraction that allows adding custom datasets such as this FLamby example.

Please follow the node configuration documentation to learn about how to instantiate a node component. After your node is ready, go to `<node-path>/data` and create a JSON file that describes your dataset. This JSON file should contain the path to the dataset (where it is downloaded using the FLamby dataset download script), and the center number that will be used to extract samples reserved for the specific FL node/center: 

```json
{
    "data-path": "<absolute-path-to-data>",
    "center": 0
}
```

After the JSON file is ready, go to the Fed-BioMed CLI and add the custom dataset:  

```shell
fedbiomed node -p <path-to-node-component> dataset add 
```

Please select the option `custom` and follow the interactive CLI to complete dataset addition. This part is exactly the same dataset deployment steps described in the dataset deployment user guide. **In the section where the data path is asked, please enter the JSON file path.**


## Defining the Training Plan

In Fed-BioMed, researchers create a [training plan](../../user-guide/researcher/training-plan.md) 
to define various aspects of their federated ML experiment, such as the model, the data, the optimizer, and others.
To leverage FLamby functionalities within your Fed-BioMed experiment, you will be required to create a 
custom training plan inheriting from `fedbiomed.common.training_plans.TorchTrainingPlan`. 

For details on the meaning of the different functions in a training plan, and how to implement them correctly, 
please follow the [TrainingPlan](../../user-guide/researcher/training-plan.md) user guide. 
Since FLamby is highly compatible with PyTorch, you may use the models and optimizers provided by FLamby in your 
Fed-BioMed experiment seamlessly. See the code below for an example:

```python
from fedbiomed.common.training_plans import TorchTrainingPlan
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

Fed-BioMed provides a (`CustomDataset`)[../../user-guide/datasets/custom-dataset.md] class that can be used for integrating and using FLamby datasets in Fed-BioMed. This class requires to be extended by providing special methods such as `read`: to read the dataset, and `get_item` to get a single item from the datast. You need to create a custom FLamby dataset  class in your training plan and use it int the `training_data` function as it is shown in the example below:

```python
from fedbiomed.common.training_plans import TorchTrainingPlan
from fedbiomed.common.datasets import CustomDataset
from fedbiomed.common.datamanager import DataManager
from flamby.datasets.fed_heart_disease import FedHeartDisease

class MyTrainingPlan(TorchTrainingPlan):
    def init_dependencies(self):
        return [
            "from flamby.datasets.fed_heart_disease import FedHeartDisease",
            "from fedbiomed.common.datamanager import DataManager",
            "from fedbiomed.common.datasets import CustomDataset",
        ]

    class MyFlambyDataset(CustomDataset):

        def read(self):
            """Read FLamby data"""            
            import json
            with open(self.path) as f:
                file - f.read()
                flamby_data = json.loads(file)
            
            self.data = FedHeartDisease(
                center=flamby_data["center"], 
                data_path=flamby_data["data-path"]
            )

        def get_item(self, item):
            """Get item"""
            return self.data[item] 

    def training_data(self):
        dataset = self.MyFlambyDataset()
        loader_arguments = {'shuffle': True}
        return DataManager(dataset, **loader_arguments)

    # ... Implement the other functions as needed ...
```

## Data Transformations

Functional data transformations can be specified in the `training_data` function, similarly to the common `TorchTrainingPlan` pattern. However, for FLamby you are required to use the special function `init_transform` of the FLamby dataset, as per the example below.

```python
from fedbiomed.common.training_plans import TorchTrainingPlan
from fedbiomed.common.datasets import CustomDataset
from fedbiomed.common.datamanager import DataManager
from monai.transforms import Compose, Resize, NormalizeIntensity

class MyTrainingPlan(TorchTrainingPlan):
    def init_dependencies(self):
        return [
            "from fedbiomed.common.datasets import CustomDataset", 
            "from fedbiomed.common.datamanager import DataManager",
            "from monai.transforms import Compose, Resize, NormalizeIntensity",
        ]
    
    class MyFlambyDataset(CustomDataset):
        def read(self):
            """Read FLamby data"""            
            from jsonlib import json
            flamby_data = json.loads(self.path)
            self.data = FedHeartDisease(
                center=flamby_data["center"], 
                data_path=flamby_data["data-path"])

        def get_item(self, item):
            """Get item"""
            return self.data[item] 

    def training_data(self):
        dataset = self.MyFlambyDataset()
        myComposedTransform = Compose([Resize((48,60,48)), NormalizeIntensity()])
        dataset.data.init_transform(myComposedTransform)

        train_kwargs = {'shuffle': True}
        return DataManager(dataset, **train_kwargs)

    # ... Implement the other functions as needed ...
```

!!! info "Transforms must always be of `Compose` type"
    Transforms added to a FLamby dataset must always be either of type `torch.transforms.Compose` or
    `monai.transforms.Compose`

Do not forget to always add your transforms as dependencies in the `init_dependencies` function!

## Next Steps

Please check the FLamby tutorials to see training examples with FLamby datasets and modules. 
