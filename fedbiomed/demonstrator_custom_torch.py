from torch.utils.data import DataLoader, Dataset
import torch

from fedbiomed.common.data._framework_native_dataset import FrameworkNativeDataset, PytorchNativeDataset

from fedbiomed.common.constants import TrainingPlans
from fedbiomed.common.data._data_manager import DataManager

class CustomPytorchDataset(Dataset):
    def __init__(self):
        self._data = torch.randn((100, 20, 20))
        self._target = torch.randint(12, (100, 1))


    def __getitem__(self, idx):
        return self._data[idx], self._target[idx]
        #return {'data': self._data[idx]}, {'target': self._target[idx]}
    
    def __len__(self):
        return len(self._target)
    
# FIXME: should the user return a dict in the `__getitem__` method?
# or should we do it automatically in `DataManager`?

# FIXME 2: should we mae it compatible with sklearn?

dataset = CustomPytorchDataset()
dataset = PytorchNativeDataset(dataset)
data_manager = DataManager(dataset)

data_manager.load(tp_type=TrainingPlans.TorchTrainingPlan)

train, test = data_manager.split(test_ratio=0.1, test_batch_size=None)
val = next(iter(train.dataset))

print(val)