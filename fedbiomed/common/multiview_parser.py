import pandas as pd
import numpy as np
from typing import Any, Dict, List, Union


class MultiViewCSVParser:
    def __init__(self,
                  dataset: pd.DataFrame,
                  is_multi_view: bool,
                  views: List[int] = None):
        self.dataset = dataset
        self._is_muti_view = is_multi_view
        if is_multi_view:
            #self.__getitem__ = self.pandas_multiview_handler
            view_names = sorted(set(dataset.columns.get_level_values(0)))
            self.view_names = list(view_names)
            self.views = []
            for view_name in view_names:
                
                self.views.append(dataset[view_name].shape[0])
            # self.views_names_map = {}
        else:
            #self.__getitem__ = self.pandas_singleview_handler
            self.views = views
            self.view_names = None

    def create_iterator(self) -> Union[list[int], List[str]]:
        if self._is_muti_view:
            return self.view_names
        else:
            return range(self.views)

    @staticmethod
    def read_multi_view_dataframe(file_name: str) -> pd.DataFrame:
        df = pd.read_csv(file_name, delimiter=',', index_col=0, header=[0,1])
        return df

    @staticmethod
    def create_multi_view_dataframe(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        _header_labels = ['views', 'feature_name']
        # 1. create multiindex header
        
        _feature_name_array = np.array([])  # store all feature names
        _view_name_array = []  # store all views (ie modalities) names
        
        _concatenated_datasets = np.array([])  # store dataframe values
        
        for key in datasets.keys():
            #_sub_dataframe_header.append(list(datasets[key].columns.values))
            _feature_name_array = np.concatenate([_feature_name_array,
                                                  datasets[key].columns.values])
            if len(_concatenated_datasets) <= 0:
                # first pass 
                _concatenated_datasets = datasets[key].values
            else:
                # next passes
                try:
                    _concatenated_datasets = np.concatenate(
                                            [_concatenated_datasets,
                                             datasets[key].to_numpy()
                                             ], axis=1)
                except ValueError as val_err:
                    # catching case where nb_samples are differents
                    raise ValueError(
                        'Cannot create multi view dataset: different number of samples for each modality have been detected'\
                            + 'Details: ' + str(val_err)
                        )
            for _ in datasets[key].columns.values:
                _view_name_array.append(key)

        _header = pd.MultiIndex.from_arrays([_view_name_array,
                                             _feature_name_array],
                                            names=_header_labels)

        
        # 2. create multi index dataframe
        
        multi_view_df = pd.DataFrame(_concatenated_datasets,
                                      columns = _header)
        return multi_view_df

    def pandas_multiview_handler(self, x: str):
        return self.dataset[x]

    def pandas_singleview_handler(self, x: int):
        if x - 1 >= 0:
            ind = self.view[x-1]
        else:
            ind = 0
        df = self.dataset.iloc[:, ind:ind + self.dim_views[x]]
        return df
    
    def __getitem__(self, x: Union[int, str]):
        if self._is_muti_view:
            return self.pandas_multiview_handler(x)
        else:
            return self.pandas_singleview_handler(x)


class MultiViewParams(dict):
    class MultiViewParam:
        def __init__(self, params: Union[List, dict]):
            self._params = params
            if isinstance(self._params, dict):
                self._key_mapping = {k:v for k, v in enumerate(self._params.keys())}
            elif isinstance(self._params, list):
                self._key_mapping = None
        def __getitem__(self, x: Union[str, int]):
            if self._key_mapping is not None:
                if isinstance(x, int):
                    # case where an iteger is used 
                    # instead of a string
                    value = self._key_mapping(x)
            else:
                if isinstance(x, str):
                  pass  
                value =  self._params[x]
            return value
    def __init__(self, params: Dict[str, Any]):
        self._params = params
        
