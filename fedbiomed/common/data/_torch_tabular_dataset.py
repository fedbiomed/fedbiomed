import numpy as np
import pandas as pd
from typing import Union
from torch import from_numpy
from torch.utils.data import Dataset
from fedbiomed.common.exceptions import FedbiomedTorchTabularDatasetError
from fedbiomed.common.constants import ErrorNumbers


class TorchTabularDataset(Dataset):

    def __init__(self,
                 inputs: Union[np.ndarray, pd.DataFrame, pd.Series],
                 target: Union[np.ndarray, pd.DataFrame, pd.Series], ):
        """
        Torch based Dataset object to create torch Dataset from given numpy or dataframe
        type of input and target variables

        Args:

            inputs (np.ndarray, pd.DataFrame, pd.Series): Input variables that will be passed to network
            target (np.ndarray, pd.DataFrame, pd.Series): Target variable for output layer

        Raises:
            FedbiomedTorchDatasetError: If input variables and target variable does not have
                                        equal length/size
        """

        # The lengths should be equal
        if len(inputs) != len(target):
            raise FedbiomedTorchTabularDatasetError(f"{ErrorNumbers.FB610.value}: Length of input variables and target "
                                                    f"variable does not match. Please make sure that they have "
                                                    f"equal size while creating the method `training_data` of "
                                                    f"TrainingPlan")

        # Inputs and target variable should be converted to the torch tensors
        # PyTorch provides `from_numpy` function to convert numpy arrays to
        # torch tensor. Therefore, if the arguments `inputs` and `target` are
        # instance one of `pd.DataFrame` or `pd.Series`, they should be converted to
        # numpy arrays
        if isinstance(inputs, (pd.DataFrame, pd.Series)):
            self.inputs = inputs.to_numpy()
        elif isinstance(inputs, np.ndarray):
            self.inputs = inputs
        else:
            raise FedbiomedTorchTabularDatasetError(f"{ErrorNumbers.FB610.value}: The argument `inputs` should be "
                                                    f"an instance one of np.ndarray, pd.DataFrame or pd.Series")
        # Configuring self.target attribute
        if isinstance(target, (pd.DataFrame, pd.Series)):
            self.target = target.to_numpy()
        elif isinstance(inputs, np.ndarray):
            self.target = target
        else:
            raise FedbiomedTorchTabularDatasetError(f"{ErrorNumbers.FB610.value}: The argument `target` should be "
                                                    f"an instance one of np.ndarray, pd.DataFrame or pd.Series")

        # Convert `inputs` adn `target` to Torch floats
        self.inputs = from_numpy(self.inputs).float()
        self.target = from_numpy(self.target).float()

    def __len__(self):
        """
        Mandatory method for pytorch Dataset. It is used for pytorch DataLoader and Fed-BioMed
        DataManager to find out number of samples and doing train/test split
        """
        return len(self.inputs)

    def __getitem__(self, item):
        """
        Mandatory method for pytorch Dataset to get input and target instance by
        given index. Used by DataLoader in training_routine to load samples by index.
        """
        return self.inputs[item], self.target[item]
