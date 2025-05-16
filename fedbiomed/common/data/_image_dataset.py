

#from fedbiomed.common.data._torch_data_manager import TorchDataManager
#from fedbiomed.common.data.converter_utils import from_torch_dataset_to_generic

import torch
class ImageDataset:
    """ImageFolder wrapper"""
    def __init__(self, dataset):
        self._dataset = dataset
        self._transform_framework = lambda x:x

    def to_torch(self):
        pass


    def to_sklearn(self):

        class ToNumpy(torch.nn.Module):
            def forward(self, img, label):
                # Do some transformations
                return img.numpy(), label
            
        if self._dataset.transforms is not None:
            self._dataset.append(ToNumpy())
        else:
            self._dataset = [ToNumpy()]
        #self._transform_framework = from_torch_dataset_to_generic
        # TODO: implement here, decide if we should use transform or collate_fn


    # def split(self, test_ratio):
    #     # implement here method where we split between testing and training dataset
    #     pass


    def to_data_manager(self, **loader_arguments):
        pass
        # return TorchDataManager(self._dataset, **loader_arguments)

class ImageDataLoader:
    pass