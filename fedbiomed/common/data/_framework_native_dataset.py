

#from fedbiomed.common.data._torch_data_manager import TorchDataManager
#from fedbiomed.common.data.converter_utils import from_torch_dataset_to_generic

import copy
from typing import Callable
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from fedbiomed.common.constants import TrainingPlans
from torchvision import  transforms

from fedbiomed.common.data._generic_dataset import GenericDataset

# class NativeDataManager:

#     def __init__(self, dataset, targets=None, **kwargs):
        
#         self._dataset = dataset
#         self._targets = targets


class FrameworkNativeDataset(GenericDataset):
    """ImageFolder wrapper"""
    def __init__(self, dataset):
        self._dataset = dataset
        self._transformed_dataset = None
        self._dataloader = None
        self._transform_framework = lambda x:x


    def to_torch(self):
        pass


    def to_sklearn(self):
        pass

    def set_dataloader(self, inputs, target=None, kwargs={}):
        pass

class PytorchNativeDataset(FrameworkNativeDataset):
    def __init__(self, dataset: Dataset):
        self._dataset = dataset
        self._collate_fn = lambda x:x[0] if isinstance(x, list) else x
        self._dataloader = DataLoader

    def __getitem__(self, idx):
        input_data, targets = self._dataset[idx]
        return {'data': input_data, 'target': targets}
    def __len__(self):
        return len(self._dataset)
    def set_dataloader(self, inputs, target=None, kwargs={}):
        if 'collate_fn' not in kwargs:
            kwargs['collate_fn'] = self._collate_fn
            # avoid calling  default collate_fn function that will convert everyting to Pytroch

        return self._dataloader(inputs,  **kwargs), self._dataloader#**kwargs)
    # @classmethod
    # def load(clf, dataset, tp_type):
    #     dataset = clf(dataset)
    #     if tp_type == TrainingPlans.TorchTrainingPlan:
    #         dataset.to_torch()
    #     elif tp_type == TrainingPlans.SkLearnTrainingPlan:
    #         dataset.to_sklearn()
    #     return clf

            
    def to_sklearn(self):
        
        class ToNumpy(torch.nn.Module):
            def forward(self, img, label=None):
                # Do some transformations

                return img.numpy() if hasattr(img, 'numpy') else np.array(img)
        
        
        if self._dataset.transforms is not None:
            self._dataset.transform.transforms.append(ToNumpy())
        else:
            self._dataset.transform= transforms.Compose([ToNumpy()])
        #self._transform_framework = from_torch_dataset_to_generic
        # TODO: implement here, decide if we should use transform or collate_fn


    # def split(self, test_ratio):
    #     # implement here method where we split between testing and training dataset
    #     pass

    def to_torch(self):
        pass

        # return TorchDataManager(self._dataset, **loader_arguments)

