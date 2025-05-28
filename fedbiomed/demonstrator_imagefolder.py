from torchvision import datasets, transforms

from fedbiomed.common.constants import TrainingPlans
from fedbiomed.common.data._data_manager import DataManager
from fedbiomed.common.data._framework_native_dataset import FrameworkNativeDataset, PytorchNativeDataset


# FIXME: in pytorch, if the `transforms.ToTorch` is not defined, returned object is a PIL image. Should
# we provide same behaviour for numpy?


# POC for MedNIST dataset: 

## 1. with tramsforms defined for pytorch
### ---- defined in the training plan
path = 'dataset/MedNIST'
preprocess = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
           ])
    
train_data = datasets.ImageFolder(path, transform = preprocess)  # object from pytorch

ds = PytorchNativeDataset(train_data)
data_manager = DataManager(dataset=ds)
###### ----------------------------- end of the training plan


data_manager.load(tp_type=TrainingPlans.TorchTrainingPlan)
val = next(iter(data_manager.dataset))
print(val)


## 2. with transform defined for sklearn
### ---- defined in the training plan
preprocess = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
           ])
train_data = datasets.ImageFolder(path, transform = preprocess)  # object from pytorch
data_manager = DataManager(dataset=ds)
data_manager.load(tp_type=TrainingPlans.SkLearnTrainingPlan)
data_manager.split(test_ratio=.0, test_batch_size=None)
val = next(iter(data_manager.dataset))
print(val)


# without transforms

## start of training plan ----------------
path = 'dataset/MedNIST'


    
train_data = datasets.ImageFolder(path,)  # object from pytorch

ds = PytorchNativeDataset(train_data)
data_manager = DataManager(dataset=ds)

## end of trainingplan ---------------------
data_manager.load(tp_type=TrainingPlans.TorchTrainingPlan)
val = next(iter(data_manager.dataset))
print(val)

data_manager = DataManager(dataset=ds)
data_manager.load(tp_type=TrainingPlans.SkLearnTrainingPlan)
val = next(iter(data_manager.dataset))
print(val)