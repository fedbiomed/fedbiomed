# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Torch tabulated data manager
"""

from typing import Optional, Union, Tuple

import numpy as np
import pandas as pd

from torch import from_numpy, stack, Tensor
from torch.utils.data import Dataset

from fedbiomed.common.exceptions import FedbiomedDatasetError
from fedbiomed.common.constants import ErrorNumbers, DatasetTypes


class TabularDataset(Dataset):
    """Torch based Dataset object to create torch Dataset from given numpy or dataframe
    type of input and target variables
    """
    def __init__(self,
                 inputs: Optional[Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
                 target: Optional[Union[np.ndarray, pd.DataFrame, pd.Series]] = None):
        """Constructs PyTorch dataset object

        Args:
            inputs: Input variables that will be passed to network
            target: Target variable for output layer

        Raises:
            FedbiomedTorchDatasetError: If input variables and target variable does not have
                equal length/size
        """

        # Inputs and target variable should be converted to the torch tensors
        # PyTorch provides `from_numpy` function to convert numpy arrays to
        # torch tensor. Therefore, if the arguments `inputs` and `target` are
        # instance one of `pd.DataFrame` or `pd.Series`, they should be converted to
        # numpy arrays
        if inputs is None:
            return
        if isinstance(inputs, (pd.DataFrame, pd.Series)):
            self.inputs = inputs.to_numpy()
        elif isinstance(inputs, np.ndarray):
            self.inputs = inputs
        else:
            raise FedbiomedDatasetError(f"{ErrorNumbers.FB610.value}: The argument `inputs` should be "
                                                    f"an instance one of np.ndarray, pd.DataFrame or pd.Series")

        self.inputs = from_numpy(self.inputs).float()

        # Configuring self.target attribute
        if isinstance(target, (pd.DataFrame, pd.Series)):
            self.target = target.to_numpy()
        elif isinstance(inputs, (np.ndarray, type(None))):
            self.target = target
        else:
            raise FedbiomedDatasetError(f"{ErrorNumbers.FB610.value}: The argument `target` should be "
                                                    f"an instance one of np.ndarray, pd.DataFrame or pd.Series")

        if self.target is not None:
            self.target = from_numpy(self.target).float()
            # The lengths should be equal
            if len(self.inputs) != len(self.target):
                raise FedbiomedDatasetError(f"{ErrorNumbers.FB610.value}: Length of input variables and target "
                                                        f"variable does not match. Please make sure that they have "
                                                        f"equal size while creating the method `training_data` of "
                                                        f"TrainingPlan")

    def __len__(self) -> int:
        """Gets sample size of dataset.

        Mandatory method for pytorch Dataset. It is used for pytorch DataLoader and Fed-BioMed
        DataManager to find out number of samples and doing train/test split

        Returns:
            Total number of samples
        """
        return len(self.inputs)

    def __getitem__(self, item: int) -> Tuple[Tensor, Tensor]:
        """ Mandatory method for pytorch Dataset to get input and target instance by
        given index. Used by DataLoader in training_routine to load samples by index.

        Args:
            item: Index to select single sample from dataset

        Returns:
            inputs: Input sample
            target: Target sample
        """
        if self.target is not None:
            return self.inputs[item], self.target[item]
        else:
            return self.inputs[item], None

    def mean(self):
        if self.target is not None:
            return {'inputs': self.inputs.mean(axis=0).tolist(),
                    'targets': self.target.mean(axis=0).tolist(),
                    'num_samples': self.target.shape[0]}
        else:
            return {'inputs': self.inputs.mean(axis=0).tolist(),
                    'targets': None,
                    'num_samples': self.target.shape[0]}

    @staticmethod
    def aggregate_mean(node_means: list):
        from functools import reduce
        total_num_samples = reduce(lambda x, y: x + y['num_samples'], node_means, 0)
        try:
            inputs = stack(
                [Tensor(x['inputs'])*x['num_samples'] for x in node_means]
            )
        except TypeError:
            # try to catch case where we have a single feature
            inputs = stack(
                [Tensor([x['inputs']])*x['num_samples'] for x in node_means]
            )
        agg_inputs_means = inputs.sum(axis=0)/total_num_samples
        if all([x['targets'] is not None for x in node_means]):
            try:
                targets = stack(
                    [Tensor(x['targets'])*x['num_samples'] for x in node_means]
                )
            except TypeError:
                # try to catch case where target is 1-dimensional
                targets = stack(
                    [Tensor([x['targets']])*x['num_samples'] for x in node_means]
                )
            agg_targets_mean = targets.sum(axis=0)/total_num_samples
        else:
            agg_targets_mean = None
        return {
            'inputs':  agg_inputs_means,
            'targets': agg_targets_mean
        }

    @staticmethod
    def get_dataset_type() -> DatasetTypes:
        return DatasetTypes.TABULAR
