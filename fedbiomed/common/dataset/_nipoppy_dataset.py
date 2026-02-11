# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union, Type, List

import json
import warnings
from functools import cached_property
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import torch

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.dataset._dataset import Dataset
from fedbiomed.common.dataset_types import DataReturnFormat
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.common.logger import logger
from nipoppy import NipoppyDataRetriever
from sklearn.model_selection import StratifiedKFold


class NipoppyDataset(Dataset):

    # columns
    COL_PARTICIPANT_ID = "participant_id"
    COL_SESSION_ID = "session_id"
    TERMURL_AGE = "nb:Age"
    TERMURL_SEX = "nb:Sex"
    TERMURL_COG_DECLINE = "fl:cognitive_decline_status"
    TERMURL_COG_DECLINE_AVAILABILITY = "fl:cognitive_decline_availability"
    TERMURL_DIAGNOSIS = "nb:Diagnosis"

    # values
    TERMURL_AVAILABLE = "nb:available"
    TERMURL_UNAVAILABLE = "nb:unavailable"
    TERMURL_MALE = "snomed:248153007"
    TERMURL_FEMALE = "snomed:248152002"
    TERMURL_HEALTHY_CONTROL = "ncit:C94342"

    # for derivatives specs
    FS_NAME = "freesurfer"
    FS_VERSION = "7.3.2"
    FS_STATS_NAME = "fs_stats"
    FS_STATS_VERSION = "0.2.1"
    SUFFIX_APARC = "-aparc.DKTatlas-thickness.tsv"
    SUFFIX_ASEG = "-aseg-volume.tsv"

    def __init__(
        self,
        phenotypes,
        derivatives,
        transforms,
        n_splits,
        i_split,
        target,
        random_state,
        fname_stats,
        train,
        null
        #...
    ) -> None:
        """
        """
        self.phenotypes = phenotypes
        self.derivatives = derivatives
        self.transforms = transforms
        self.n_splits = n_splits
        self.i_split = i_split
        self.target = target
        self.random_state = random_state
        self.fname_stats = fname_stats
        self.train = train
        self.null = null
    
    def complete_initialization(
        self, controller_kwargs: Dict[str, Any], to_format: DataReturnFormat
    ) -> None:
        """Finalize initialization of object to be able to recover items

        Args:
            path: path to dataset
            to_format: format associated to expected return format
        """

        self.path = controller_kwargs.get("root", None)
        if self.path is None:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Custom Dataset ERROR: 'root' must be provided in controller_kwargs to specify dataset location."
            )
        self._to_format = to_format

        # Call user defined read function to read the dataset
        try:
            self.read()
        except Exception as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Failed to read "
                f"from dataset using read method. Please see error: {e}"
            ) from e

        try:
            sample = self[0]
        except Exception as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Failed to retrieve item "
                f"from dataset using get_item method. Please see error: {e}"
            ) from e
        if not isinstance(sample, tuple) or len(sample) != 2:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: get_item method must return a tuple of two elements"
                f" (data, target), but got {type(sample).__name__} with"
                f" length {len(sample) if isinstance(sample, (list, tuple)) else 'N/A'}"
            )

        # Following line is just to check that dataset is well implemented
        # and it return correct data type respecting to to_format
        try:
            sample = self[0]
        except Exception as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Failed to retrieve item "
                f"from dataset using get_item method. Please see error: {e}"
            ) from e
    
    @cached_property
    def config(self) -> dict:
        if Path(self.path).is_file():
            dpath_parent = Path(self.path).parent
        else:
            dpath_parent = Path(self.path)
        return json.loads((dpath_parent / "global_config.json").read_text())[
            "CUSTOM"
        ]["FL_PD"]

    def filter_sessions(self, df: pd.DataFrame) -> pd.DataFrame:
        if (session_id := self.config.get("SINGLE_SESSION", None)) is not None:
            df = df.query(
                f"{self.COL_SESSION_ID} == '{session_id}'"
            )
        elif (
            mapping_file_path := self.config.get("MAPPING_FILE", None)
        ) is not None:
            df_mapping = pd.read_csv(
                Path(self.path, mapping_file_path),
                sep="\t",
                header=None,
                names=[
                    self.COL_PARTICIPANT_ID,
                    self.COL_SESSION_ID,
                ],
                dtype=str,
            )
            idx = pd.MultiIndex.from_frame(df_mapping)
            df = df.loc[idx]

        # make sure participant IDs are unique
        participant_ids = df.index.get_level_values(
            self.COL_PARTICIPANT_ID
        )
        if len(participant_ids) != len(set(participant_ids)):
            raise ValueError("Some participants have more than one session")

        return df

    def standardize_df(
        self, df: pd.DataFrame, cols_to_ignore=None
    ) -> pd.DataFrame:
        if cols_to_ignore is None:
            cols_to_ignore = []
        cols_to_ignore.append("dataset")

        fpath_stats = Path(self.config["STATS"], self.fname_stats)
        df_stats = pd.read_csv(fpath_stats, sep="\t", index_col=0)
        for col in df.columns:
            if col in cols_to_ignore:
                continue
            if col not in df_stats.columns:
                raise ValueError(f"{col=} not in {fpath_stats=}")
            mean = df_stats.at["mean", col]
            std = df_stats.at["std", col]
            if std == 0:
                raise ValueError(f"std is zero for {col=} in {fpath_stats=}")
            df[col] = (df[col] - mean) / std
        return df

    def read(self):

        if Path(self.path).is_file():
            df = pd.read_csv(
                self.path,
                sep="\t",
                dtype={
                    self.COL_PARTICIPANT_ID: str,
                    self.COL_SESSION_ID: str,
                },
            ).set_index(
                [
                    self.COL_PARTICIPANT_ID,
                    self.COL_SESSION_ID,
                ]
            )
        else:
            retriever = NipoppyDataRetriever(self.path)
            try:
                df = retriever.get_tabular_data(
                    phenotypes=self.phenotypes,
                    derivatives=self.derivatives,
                )
            except Exception as e:
                logger.warning(
                    f"Error retrieving data from {self.path}: {e}. Trying again without diagnosis column."
                )
                df = retriever.get_tabular_data(
                    phenotypes=[
                        phenotype
                        for phenotype in self.phenotypes
                        if phenotype != self.TERMURL_DIAGNOSIS
                    ],
                    derivatives=self.derivatives,
                )

            # filter sessions
            df = self.filter_sessions(df)

        # for building mega dataset
        self.df_before_transforms: pd.DataFrame = df.copy()

        # apply transforms
        for transform in self.transforms:
            df = transform(df)
            if not isinstance(df, pd.DataFrame):
                raise TypeError(
                    "Transform functions must return a pandas DataFrame."
                )

        # after transforms but before splits
        self.df_after_transforms: pd.DataFrame = df.copy()

        # stratification variable
        if self.target == self.TERMURL_AGE:
            bins = np.arange(0, 100, 5)
            y = pd.cut(df[self.target], bins=bins).astype(str)
        else:
            y = df[self.target]

        # get CV fold
        cv = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )
        splits = cv.split(np.arange(len(df)), y=y)
        for _ in range(self.i_split + 1):
            idx_train, idx_test = next(splits)
        if self.train:
            idx = idx_train
        else:
            idx = idx_test
        df = df.iloc[idx]

        if self.fname_stats is not None:
            df = self.standardize_df(df, cols_to_ignore=[self.target])

        self.df: pd.DataFrame = df.copy()
        self.y = self.df[[self.target]]
        self.X = self.df.drop(labels=[self.target], axis="columns")

        if self.null:
            rng = np.random.default_rng()
            idx = np.arange(len(self.y))
            rng.shuffle(idx)
            self.y = self.y.iloc[idx]

        return self.X, self.y

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray | None]:
        return self.X.iloc[idx].to_numpy(), self.y.iloc[idx].to_numpy()
