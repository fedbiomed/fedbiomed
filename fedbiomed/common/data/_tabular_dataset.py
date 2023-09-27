# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Torch tabulated data manager
"""

from typing import Optional, Union, Tuple

import numpy as np
import pandas as pd
from functools import reduce
from collections import OrderedDict

import torch
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
            return {'inputs': self.inputs.mean(axis=0),
                    'targets': self.target.mean(axis=0),
                    'num_samples': self.target.shape[0]}
        else:
            return {'inputs': self.inputs.mean(axis=0),
                    'targets': None,
                    'num_samples': self.target.shape[0]}

    @staticmethod
    def flatten_mean(means: dict):
        format = OrderedDict()
        inputs_means = means.pop('inputs')
        format['inputs'] = list(inputs_means.shape) or [1]
        inputs_means = inputs_means.flatten()
        targets_means = means.pop('targets')
        format['targets'] = list(targets_means.shape) or [1]
        targets_means = targets_means.flatten()
        return {'flat': torch.cat((inputs_means, targets_means), dim=0),
                'format': format,
                **means}

    @staticmethod
    def _unflatten(flat_results: dict):
        out = {}
        offset = 0
        for key, shape in flat_results['format'].items():
            numel = reduce(lambda x, y: x * y, shape, 1)
            t = flat_results['flat'][offset:offset + numel]
            out[key] = torch.Tensor(t).reshape(shape)
            offset += numel
        return out

    def unflatten_mean(flat_results: dict):
        return TabularDataset._unflatten(flat_results)

    @staticmethod
    def aggregate_mean(node_means: list):
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

    def std(self, fed_mean=None):
        if self.target is not None:
            return {'inputs': {
                        'local_std': self.inputs.std(axis=0),
                        'fed_sum_of_squares': np.power(
                            (self.inputs - fed_mean['inputs']), 2
                        ).sum(axis=0)
                    },
                    'targets': {
                        'local_std': self.target.std(axis=0),
                        'fed_sum_of_squares': np.power(
                            (self.target - fed_mean['targets']), 2
                        ).sum(axis=0)
                    },
                    'num_samples': self.target.shape[0]}
        else:
            return {'inputs': {
                        'local_std': self.inputs.std(axis=0),
                        'fed_sum_of_squares': np.power(
                            (self.inputs - fed_mean['inputs']), 2
                        ).sum(axis=0)
                    },
                    'targets': None,
                    'num_samples': self.target.shape[0]}

    @staticmethod
    def aggregate_std(node_results: list):
        total_num_samples = reduce(lambda x, y: x + y['num_samples'], node_results, 0)
        total_ss_inputs = reduce(lambda x, y: x + y['inputs']['fed_sum_of_squares'], node_results, 0)
        agg_inputs_std = np.sqrt(total_ss_inputs/(total_num_samples-1))
        if all([x['targets'] is not None for x in node_results]):
            total_ss_targets = reduce(lambda x, y: x + y['targets']['fed_sum_of_squares'], node_results, 0)
            agg_targets_std = np.sqrt(total_ss_targets/(total_num_samples-1))
        else:
            agg_targets_std = None
        return {
            'inputs':  agg_inputs_std,
            'targets': agg_targets_std
        }

    @staticmethod
    def flatten_std(stds: dict):
        """

        inputs are discarded.
        """
        format = OrderedDict()
        ssq_stds = stds.pop('fed_sum_of_squares')
        format['fed_sum_of_squares'] = list(ssq_stds.shape) or [1]
        ssq_stds = ssq_stds.flatten()
        targets_stds = stds.pop('targets')
        format['targets'] = list(targets_stds.shape) or [1]
        targets_stds = targets_stds.flatten()
        return {'flat': torch.cat((inputs_stds, ssq_stds, targets_stds), dim=0),
                'format': format,
                **stds}

    def unflatten_std(flat_results: dict):
        return TabularDataset._unflatten(flat_results)

    @staticmethod
    def get_dataset_type() -> DatasetTypes:
        return DatasetTypes.TABULAR
