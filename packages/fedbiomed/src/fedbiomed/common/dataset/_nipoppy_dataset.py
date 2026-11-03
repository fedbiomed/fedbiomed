# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.dataloadingplan import DataLoadingPlan
from fedbiomed.common.dataset._dataset import Dataset
from fedbiomed.common.dataset_controller._nipoppy_controller import (
    Derivatives,
    InstalledPipelines,
    NipoppyController,
    SessionFilters,
)
from fedbiomed.common.dataset_types import DataReturnFormat
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedValueError
from fedbiomed.common.logger import logger


class NipoppyDataset(Dataset):
    """Dataset for tabular data (phenotypes and derivatives) from a Nipoppy study.

    Each sample is one row of the study, i.e. one `(participant_id, session_id)`
    pair. Input features are all selected columns except `target`.

    Typical lifecycle:
    1. instantiated in the training plan's `training_data`;
    2. `load` is called by the node, which creates the controller, runs the
       optional derivatives initializer, reads the data and validates the
       transforms on the first sample;
    3. samples are retrieved with `__getitem__`.
    """

    _controller_cls: type = NipoppyController
    _controller: NipoppyController

    _native_to_framework = {
        DataReturnFormat.SKLEARN: lambda x: x.to_numpy(),
        DataReturnFormat.TORCH: lambda x: torch.from_numpy(x.to_numpy()),
    }

    # columns
    COL_PARTICIPANT_ID = NipoppyController.COL_PARTICIPANT_ID
    COL_SESSION_ID = NipoppyController.COL_SESSION_ID

    def __init__(
        self,
        phenotypes: Optional[Union[str, List[str]]] = None,
        derivatives: Optional[Derivatives] = None,
        target: Optional[Union[str, List[str]]] = None,
        session_filters: Optional[SessionFilters] = None,
        drop_na: bool = True,
        drop_na_kwargs: Optional[Dict[str, Any]] = None,
        sample_level_transform: Optional[Callable] = None,
        sample_level_target_transform: Optional[Callable] = None,
        whole_df_level_transform: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
        derivatives_initializer: Optional[Callable[["NipoppyDataset"], None]] = None,
    ) -> None:
        """Constructor of the class

        Args:
            phenotypes: Phenotypes to use as input features. If `None`, all
                phenotypes listed in `get_metadata()["available_phenotypes"]`
                are used.
            derivatives: Derivatives to use as input features, as
                `(pipeline, version, path)` tuples. Can also be set later by a
                `derivatives_initializer`.
            target: Phenotype(s) to use as target. If `None`, samples have no target.
            session_filters: Sessions to keep. Either a session id, a list of
                session ids, or a list of `(participant_id, session_id)` tuples.
            drop_na: Whether to drop rows with missing values
            drop_na_kwargs: Keyword arguments forwarded to `pd.DataFrame.dropna`.
                Ignored if `drop_na` is `False`.
            sample_level_transform: Transform applied to the input features of
                each sample, after conversion to the training plan's format
            sample_level_target_transform: Transform applied to the target of
                each sample, after conversion to the training plan's format
            whole_df_level_transform: Transform applied once to the whole
                dataframe, after session filters and `dropna`
            derivatives_initializer: Callable run by `load` after the controller
                is created and before the data is read. Signature:
                `fn(dataset: NipoppyDataset) -> None`. Typically used to set
                `dataset.derivatives` based on `dataset.get_metadata()`.

        Raises:
            FedbiomedValueError: if a transform or `derivatives_initializer` is not callable
        """
        self.phenotypes = [phenotypes] if isinstance(phenotypes, str) else phenotypes
        self.derivatives = derivatives
        self.target = [target] if isinstance(target, str) else target
        self.session_filters = session_filters

        self._transform = self._validate_transform(sample_level_transform)
        self._target_transform = self._validate_transform(sample_level_target_transform)
        self._whole_df_level_transform = whole_df_level_transform

        self._drop_na = drop_na
        self._drop_na_kwargs = drop_na_kwargs
        if not self._drop_na and drop_na_kwargs is not None:
            logger.warning(
                "drop_na is set to False but drop_na_kwargs is not None. "
                "drop_na_kwargs will be ignored."
            )

        self._derivatives_initializer: Optional[Callable[["NipoppyDataset"], None]] = None
        if derivatives_initializer is not None:
            self.set_derivatives_initializer(derivatives_initializer)

    def load(
        self,
        root: Union[str, Path],
        to_format: DataReturnFormat,
        dlp: Optional[DataLoadingPlan] = None,
        available_phenotypes: Optional[List[str]] = None,
        installed_pipelines: Optional[InstalledPipelines] = None,
    ) -> None:
        """Finalize initialization of object to be able to recover items

        Creates the controller, runs the derivatives initializer (if any), reads
        the selected data and validates the transforms on the first sample.

        Args:
            root: Root directory of the Nipoppy study
            to_format: Format of the data returned by `__getitem__`
            dlp: Data loading plan. Not supported for Nipoppy datasets; must be `None`.
            available_phenotypes: Phenotypes stored in the node database at
                registration, passed back by the node. Forwarded to the
                controller and returned as-is by `get_metadata` (`None` if not
                given).
            installed_pipelines: Pipelines stored in the node database at
                registration. Same behavior as `available_phenotypes`.

        Raises:
            FedbiomedError: if the controller cannot be created, the data cannot
                be selected, or the transforms fail on the first sample
        """
        self.to_format = to_format

        self._init_controller(
            root=root,
            session_filters=self.session_filters,
            drop_na=self._drop_na,
            drop_na_kwargs=self._drop_na_kwargs,
            whole_df_transform=self._whole_df_level_transform,
            available_phenotypes=available_phenotypes,
            installed_pipelines=installed_pipelines,
            dlp=dlp,
        )

        if self._derivatives_initializer is not None:
            try:
                self._derivatives_initializer(self)
            except Exception as e:
                raise FedbiomedError(
                    f"{ErrorNumbers.FB632.value}: Failed to run derivatives initializer. {e}"
                ) from e

        self._controller.select(
            phenotypes=self._columns_to_read(),
            derivatives=self.derivatives,
        )

        # Recover sample and validate consistency of transforms
        data, target = self._split_sample(self._controller.get_sample(0))
        self._validate_format_and_transformations(data, transform=self._transform)
        if target is not None:
            self._validate_format_and_transformations(
                target, transform=self._target_transform
            )

    def __getitem__(
        self, idx: int
    ) -> Tuple[
        Union[np.ndarray, torch.Tensor], Optional[Union[np.ndarray, torch.Tensor]]
    ]:
        """Retrieve item at index `idx`

        Args:
            idx: index of the item to retrieve

        Raises:
            FedbiomedError: if dataset has not completed initialization

        Returns:
            A tuple of input features and target (`None` if no `target`)
        """
        if self._controller is None:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: This dataset object has not completed "
                "initialization. Call `load` first."
            )

        data, target = self._split_sample(self._controller.get_sample(idx))
        sample = self.apply_transforms({"data": data, "target": target})
        return sample["data"], sample["target"]

    def _columns_to_read(self) -> List[str]:
        """Phenotype columns to read: input phenotypes followed by target columns.

        If `phenotypes` is `None`, all phenotypes in
        `get_metadata()["available_phenotypes"]` are used.

        Raises:
            FedbiomedError: if `phenotypes` is `None` and no available phenotypes
                were passed to `load`
        """
        if self.phenotypes is not None:
            phenotypes = self.phenotypes
        else:
            phenotypes = self._controller.metadata["available_phenotypes"]
            if phenotypes is None:
                raise FedbiomedError(
                    f"{ErrorNumbers.FB632.value}: `phenotypes` is None and no "
                    "`available_phenotypes` were passed to `load`. Specify "
                    "`phenotypes` explicitly."
                )
        target = self.target or []
        return list(phenotypes) + [t for t in target if t not in phenotypes]

    def _split_sample(self, sample: pd.Series) -> Tuple[pd.Series, Optional[pd.Series]]:
        """Splits a row into input features and target"""
        if self.target is None:
            return sample, None
        return sample.drop(self.target), sample[self.target]

    def get_metadata(self) -> Dict[str, Any]:
        """Returns metadata about the dataset.

        Includes available phenotypes and installed pipelines, as stored in the
        node database at registration and passed by the node to `load`. A value
        is `None` if it was not passed. Empty before `load` is called.
        """
        return self._controller.metadata if self._controller is not None else {}

    def set_derivatives_initializer(
        self, derivatives_initializer: Callable[["NipoppyDataset"], None]
    ) -> None:
        """Sets the derivatives initializer for the dataset.

        It is called by `load` after the controller is created (so that
        `get_metadata` is available) and before the data is read.

        Args:
            derivatives_initializer: Callable taking the dataset as only argument,
                e.g. to set `dataset.derivatives`. Signature:
                `fn(dataset: NipoppyDataset) -> None`.

        Raises:
            FedbiomedValueError: if `derivatives_initializer` is not callable
        """
        if not callable(derivatives_initializer):
            raise FedbiomedValueError(
                f"{ErrorNumbers.FB632.value}: `derivatives_initializer` must be callable."
            )
        self._derivatives_initializer = derivatives_initializer
