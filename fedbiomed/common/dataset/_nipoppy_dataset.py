# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Dict, Optional, Tuple
import pandas as pd
import torch

from fedbiomed.common.dataset._dataset import Dataset
from fedbiomed.common.dataset_controller._nipoppy_controller import NipoppyController
from fedbiomed.common.dataset_types import DataReturnFormat



class NipoppyDataset(Dataset):
    _controller_cls: type = NipoppyController 

    _native_to_framework = {
        DataReturnFormat.SKLEARN: lambda x: x.values,
        DataReturnFormat.TORCH: lambda x: torch.from_numpy(x.values),
    }

    # columns
    COL_PARTICIPANT_ID = NipoppyController.COL_PARTICIPANT_ID
    COL_SESSION_ID = NipoppyController.COL_SESSION_ID
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
        target,
        session_filters,
        drop_na: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        reader_transform: Optional[Callable] = None,
    ) -> None:
        """
        """
        self.phenotypes = phenotypes
        self.derivatives = derivatives
        self.transform = transform
        self.target_transform = target_transform 
        self.target = target
        self.session_filters = session_filters
        self._drop_na = drop_na
        self._reader_transform = reader_transform
    
    def complete_initialization(
        self, controller_kwargs: Dict[str, Any], to_format: DataReturnFormat
    ) -> None:
        """Finalize initialization of object to be able to recover items

        Args:
            path: path to dataset
            to_format: format associated to expected return format
        """
        controller_kwargs["session_filters"] = self.session_filters
        controller_kwargs["drop_na"] = self._drop_na
        controller_kwargs["reader_transform"] = self._reader_transform
        self._init_controller(controller_kwargs=controller_kwargs)
        self._to_format = to_format

    def __getitem__(self, idx) -> Tuple[pd.DataFrame, pd.DataFrame | None]:
        target_ = self.target if self.target is not None else []
        sample: pd.DataFrame = self._controller.get_sample(idx,
                                                           phenotypes=self.phenotypes + target_,
                                                           derivatives=self.derivatives)  # type: ignore
        Y = sample[self.target] if self.target is not None else None
        X = sample.drop(self.target) if self.target is not None else sample
        X, Y = map(self._get_format_conversion_callable(), (X, Y))
        X, Y = self._apply_transform(X, Y)
        return X, Y
    
    def _apply_transform(self, X, Y):
        if self.transform is not None:
            X = self.transform(X)
        if self.target_transform is not None and Y is not None:
            Y = self.target_transform(Y)
        return X, Y


if __name__ == "__main__":
    from fedbiomed.common.dataset_types import DataReturnFormat
    from skrub import TableVectorizer
    from sklearn.preprocessing import OneHotEncoder
    def transform_skrub(df: pd.DataFrame) -> pd.DataFrame:
        specific_transformers = []
        if NipoppyDataset.TERMURL_SEX in df.columns:
            specific_transformers.append(
                (
                    OneHotEncoder(
                        drop=[NipoppyDataset.TERMURL_MALE], sparse_output=False
                    ),
                    [NipoppyDataset.TERMURL_SEX],
                )
            )
        if NipoppyDataset.TERMURL_COG_DECLINE in df.columns:
            specific_transformers.append(
                (
                    OneHotEncoder(
                        drop=[NipoppyDataset.TERMURL_UNAVAILABLE],
                        sparse_output=False,
                        feature_name_combiner=lambda x, _: x,
                    ),
                    [NipoppyDataset.TERMURL_COG_DECLINE],
                )
            )

        table_vectorizer = TableVectorizer(specific_transformers=specific_transformers)
        df = table_vectorizer.fit_transform(df)
        return df
    dataset = NipoppyDataset(phenotypes=[
        NipoppyDataset.TERMURL_AGE,
        NipoppyDataset.TERMURL_SEX,
        NipoppyDataset.TERMURL_DIAGNOSIS,
        NipoppyDataset.TERMURL_COG_DECLINE_AVAILABILITY,
    ],
                            derivatives=[("freesurfer",
                                          "7.3.2",
                                          "idp/fs_stats-0.2.1/fs7.3.2-aparc.DKTatlas-thickness.tsv",)],
                            target=[NipoppyDataset.TERMURL_COG_DECLINE],
                            session_filters='01',  # session filter
                            drop_na=True,
                            transform=None,
                            target_transform=None,
                            reader_transform=transform_skrub)
    dataset.complete_initialization(
        controller_kwargs={"root": 
            "/Users/fcremone/dev/projects/nipoppy/nipoppy_example/my_dataset"}, 
        to_format=DataReturnFormat.SKLEARN)
    print(dataset[0])
    print(dataset[1])