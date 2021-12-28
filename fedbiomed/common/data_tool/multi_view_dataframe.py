from typing import Dict, Union, List, Tuple, Any, Optional

import pandas as pd    
import numpy as np
from fedbiomed.common.data_tool.data_type import DataType
from fedbiomed.common.data_tool import utils

# utility functions for multi view dataframe
def rename_variables_before_joining(multi_view_datasets: Dict[str, pd.DataFrame],
                                    views_name: List[Union[str, int]],
                                    primary_key:Union[str, int]=None) -> Tuple[Dict[str, pd.DataFrame],
                                                                              Dict[str, str]]:
    """
    Renames variables that have same name but different views using the following naming convention:
    if `a` is the name of a feature of `view1` and `a` is the name of a feature of `view2`,
    features names will be updated into `view1.a` and `view2.a`
    """
    _features_names = {}
    _views_length = len(views_name)
    
    # check for each variable name existing in one view, that it doesnot exist in another
    # view. if it is, rename both variables
    # for this purpose, parse every combination once
    for i_left in range(0, _views_length-1):
        _left_view = views_name[i_left]
        _left_features_name = multi_view_datasets[_left_view].columns.tolist()
        for i_right in range(i_left+1, _views_length):
        
            _right_view = views_name[i_right]
            _right_features_name = multi_view_datasets[_right_view].columns.tolist()
            
            for _f in _left_features_name:
                if primary_key and _f == primary_key:
                    # do not affect primary key (if any)
                    continue
                if _f  in _right_features_name:
                    
                    if _left_view  not in _features_names:
                        _features_names[_left_view] = {}
                        
                    if _right_view not in _features_names:
                        _features_names[_right_view] = {}
                        
                    _features_names[_left_view].update({_f: _left_view + '.' + str(_f)})
                    _features_names[_right_view].update({_f: _right_view + '.' + str(_f)})
    
    for i in range(_views_length):
        _view = views_name[i]
        _new_features = _features_names.get(_view)
        if _new_features:
            multi_view_datasets[_view] = multi_view_datasets[_view].rename(columns=_new_features)
        
    
    return multi_view_datasets, _features_names


def create_multi_view_dataframe_from_dictionary(datasets: Dict[str, pd.DataFrame],
                                                header_labels:List[str] = ['views', 'feature_name']) -> pd.DataFrame:
    # WARNING: DOESNOT CONTAIN FACILITY FOR KEEPING PRIMARY KEY
    
    # 1. create multiindex header

    _feature_name_array = np.array([])  # store all feature names
    _view_name_array = []  # store all views (ie modalities) names

    _concatenated_datasets = np.array([])  # store dataframe values

    for key in datasets.keys():
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
                                        names=header_labels)


    # 2. create multi index dataframe

    multi_view_df = pd.DataFrame(_concatenated_datasets,
                                  columns = _header)
    return multi_view_df


def create_multi_view_dataframe_from_dataframe(dataframe: pd.DataFrame,
                                               views_features: Dict[str, List[str]],
                                               primary_key: str = None):
    # convert plain dataframe into multi index dataframe
    # primary key will have its own view
    _header_labels = ['views', 'feature_name']
    _primary_key_label = 'primary_key'
    _n_features = 0
    
    #_multi_index = dataframe.columns
    if primary_key is not None:
        _key_values = dataframe[primary_key].values  # storing primary key values

    _all_features_names = []
    _new_views_names = []
    for view_name in views_features.keys():
        # get all columns name for each view, and remove primary keymulti_view_dataset[view_name] = pd.concat[]
        _features_names = list(views_features[view_name])
        
        if primary_key is not None:
            _features_names.remove(primary_key)
        
        for _ in _features_names:
            #if feature_name not in _all_features_names:
            _new_views_names.append(view_name)
            # appending as much as there are feature within each view
        _n_features += len(_features_names)
        _all_features_names.extend(_features_names)
    
    print('length', _all_features_names, _new_views_names)
    _header = pd.MultiIndex.from_arrays([ _new_views_names, _all_features_names],
                                        names=_header_labels)
    

    multi_view_dataframe = pd.DataFrame(dataframe[_all_features_names].values, columns=_header)
    
    if primary_key is not None:
        
        multi_view_dataframe[_primary_key_label, primary_key] = _key_values  # creating a specific value for
    # private key
    return multi_view_dataframe


def join_multi_view_dataset(multi_view_dataset: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
                            primary_key: str = None,
                            as_multi_index: bool = True) -> pd.DataFrame:
    """Concatenates a multi view dataset into a plain pandas dataframe,
    by doing a join operation along specified primary_key"""
    
    if isinstance(multi_view_dataset, pd.DataFrame):
        _views_names = sorted(set(multi_view_dataset.columns.get_level_values(0)))  # get views name

        
    elif isinstance(multi_view_dataset, dict):
        _views_names = sorted(list(multi_view_dataset.keys()))
        
    else:
        raise ValueError('method can only accept as input multi view pandas dataframe or dictionary of pandas dataframes')
        
    joined_dataframe = multi_view_dataset[_views_names[0]]  # retrieve the first view
    # (as a result of join operation)
    for x in range(1, len(_views_names)):
        joined_dataframe = joined_dataframe.merge(multi_view_dataset[_views_names[x]],
                                                    on=primary_key,
                                                    suffixes=('', '.'+_views_names[x]))
    
    if as_multi_index:
        # convert plain dataframe into multi index dataframe
        # primary key will have its own view
        _header_labels = ['views', 'feature_name']
        _primary_key_label = 'primary_key'

        
        _key_values = joined_dataframe[primary_key].values  # storing primary key

        _all_features_names = []
        _new_views_names = []
        for view_name in _views_names:
            # get all columns name for each view, and remove primary key
            _features_names = list(multi_view_dataset[view_name].columns)
            if primary_key is not None:
                _features_names.remove(primary_key)
            _all_features_names.extend(_features_names)

            for feature_name in _features_names:
                _new_views_names.append(view_name)
                # appending as much as there are feature within each view
            #features_name[name].remove(primary_key)

        _header = pd.MultiIndex.from_arrays([ _new_views_names, _all_features_names],
                                            names=_header_labels)
        print(_header)
        joined_dataframe  = pd.DataFrame(joined_dataframe[_all_features_names].values, columns=_header)
        joined_dataframe[_primary_key_label, primary_key] = _key_values
        
    return joined_dataframe



def search_primary_key(format_file_ref: Dict[str, Dict[str, Any]]) -> Optional[str]: 
    """[summary]

    Args:
        format_file_ref (Dict[str, Dict[str, Any]]): [description]

    Returns:
        Optional[str]: [description]
    """
    _views_names = utils.get_view_names(format_file_ref)
    primary_key = None
    _c_view = None
    for view_name in _views_names:
        file_content = format_file_ref[view_name]
        _features_names = list(file_content.keys())
        for feature_name in _features_names:
            feature_content  = file_content[feature_name]
            _d_format = feature_content.get('data_format')
            
            if _d_format == DataType.KEY.name:
                if _c_view is None:
                    primary_key = feature_name
                    _c_view = view_name
                    print(f'found primary key {primary_key}')
                else:
                    print(f'error: found 2 primary keys is same view {view_name}')
        _c_view = None
    return primary_key



def select_data_from_format_file_ref(datasets: Dict[str, Dict[str, Any]],
                                     format_file: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Returns an updated dataset containing only the features detailed in format_file"""
    # variables initialisation
    
    updated_dataset = {}
    _views_format_file = utils.get_view_names(format_file)
    
    for view in _views_format_file:
        if view in datasets.keys():
            # only extract features from format_file
            _format_file_features = list(format_file[view].keys())
            _current_dataset_feature = datasets[view].columns.tolist()
            try:
                updated_dataset[view] = datasets[view][_format_file_features]
            except KeyError as ke:
                # catch error if a column is specified in data format file
                # but not found in dataset
                _missing_feature = []
                for feature in _format_file_features:
                    if feature not in _current_dataset_feature:
                        _missing_feature.append(feature)
                print('Error: th following features', *_missing_feature, f'are not found in view: {view}')
        else:
            # trigger error
            print(f'error!: missing view {view} in dataset')
            
    return updated_dataset

def create_dictionary_multi_view_dataset(dataframe: pd.DataFrame,
                                         views_features_mapping: Dict[str, List[str]],
                                         primary_key: str=None) -> Dict[str, pd.DataFrame]:
    _primary_key_label = 'primary_key'
    
    multi_view_dataset = {}
    
    if primary_key is not None:
        _key_values = dataframe[primary_key].values  # storing primary key values

    for view_name in views_features_mapping.keys():
        # get all columns name for each view, and remove primary key
        _features_names = list(views_features_mapping[view_name])
        
        if primary_key is not None:
            _features_names.remove(primary_key)
        _tmp_dataframe = dataframe[_features_names[0]].values
        _tmp_dataframe = _tmp_dataframe.reshape(-1, 1)  # need to reshape,
        #(otherwise concatenation wont work)
        for feature in _features_names[1:]:
            # iterate over the remaining items in _feature_name
            # need to do it that way because indexing dataframe is somehow broken
            
            _new_feature = dataframe[feature].to_numpy()
            _new_feature = _new_feature.reshape(-1, 1)
            _tmp_dataframe = np.concatenate([_tmp_dataframe, _new_feature], axis=1)
            
        multi_view_dataset[view_name] =pd.DataFrame( _tmp_dataframe, columns=_features_names)
    
    if primary_key is not None:
        multi_view_dataset[primary_key] = dataframe[primary_key]
    return multi_view_dataset
