from enum import Enum
import math 

import pandas as pd
import numpy as np
from typing import List, Union, Dict

import fedbiomed.common.data_tool.utils as utils
from fedbiomed.common.data_tool.data_type import DataType
from fedbiomed.common.data_tool.pre_processing_checks  import PreProcessingChecks
from fedbiomed.common.data_tool.warning_logger import WarningReportLogger
import fedbiomed.common.data_tool.pre_processing_warnings_exceptions as check_exception
from fedbiomed.common.data_tool.data_format_ref_cli import GLOBAL_THRESHOLDS

    
def raise_warning(warning: PreProcessingChecks, *kwargs) -> str:
    if isinstance(warning.warning_type, check_exception.WarningType):
        #warning.warning_type.value(warning_disclosure)
        
        return warning.error_message % kwargs
    elif issubclass(warning.warning_type, Exception):
        
        raise warning(*kwargs)


class PreProcessingChecker:
    def __init__(self,
                 file_format_ref: dict,
                 data_frame: pd.DataFrame,
                 file_format_ref_name:str, 
                 warning_logger: WarningReportLogger):
        self._file_format_ref = file_format_ref
        self._data_frame = data_frame
        self._file_format_ref_name = file_format_ref_name
        self._warning_logger = warning_logger
        self._new_features_name = None
        #self._min_nb_samples = min_nb_samples
        self._view_feature_names = {v: [f for f in file_format_ref[v].keys()] for v in file_format_ref.keys()}
        self._features = None
        
        self._process_thresholds()  # add thresholds
        self._warning_logger.clean_report()
    
    def _process_thresholds(self):
        _thresholds_details = self._file_format_ref.get(GLOBAL_THRESHOLDS)
        if _thresholds_details is not None:
            self._min_nb_samples = _thresholds_details.get('min_nb_samples')
            self._min_nb_missing_samples = _thresholds_details.get('min_nb_missing_samples')
        else:
            self._min_nb_samples = None
            self._min_nb_missing_samples = None


    def _get_all_features(self, view:str) -> List[str]:
        return self._view_feature_names.get(view)
    
    def _get_feature_defined_in_format_file(self, view: str, feature: str)-> str:
        #feature_name = self._file_format_ref[view][feature]
        
        if self._new_features_name is not None:
            
            _new_features_name = self._new_features_name.get(view)
            
            if _new_features_name is not None and feature in _new_features_name.keys():
                feature = _new_features_name.get(feature)
                
        return feature
    
    def get_warning_logger(self):
        return self._warning_logger.get_report()
    
    def update_views_features_name(self, new_features_name: Dict[str, Dict[str, str]]):
        self._new_features_name = new_features_name
        for view in self._view_feature_names:
            if view in new_features_name:
                if view != GLOBAL_THRESHOLDS:
                    for former_feature_name, new_feature_name in new_features_name[view].items():
                        self._view_feature_names[view].append(new_feature_name)
                        self._view_feature_names[view].remove(former_feature_name)
        print('features names updated')
    
    def check_all(self, view:str=None, feature:Union[str, List[str]]=None):
        
        if view is not None:
            _views = [view]
        
        else:
            _views = utils.get_view_names(self._view_feature_names)
        
        
        for _view in _views:
            # define here test that happens on whole dataset
            
            ###
            if self._min_nb_samples is not None:
                self.check_number_of_samples(self._min_nb_samples, _view)
            
            ####
            _features = self._file_format_ref[_view].keys()
            
            for _feature in _features:
                
                # check if feature does exist
                _is_feature_exist = self.check_feature_exists_in_dataset(_view,
                                                                         _feature)
                if not _is_feature_exist:
                    continue
                
                #
                _is_format_file_correct = self.check_missing_entry_format_file_ref(_view,
                                                                                   _feature)
                if not _is_format_file_correct:
                    continue
                
                self.check_correct_variable_sub_type(_view, _feature)
                self.check_missing_values(_view, _feature)
                if self._min_nb_missing_samples is not None:
                    self.check_missing_values_threshold(_view, _feature,
                                                        self._min_nb_missing_samples)
                self.check_lower_bound(_view, _feature)
                self.check_upper_bound(_view, _feature)
                self.check_values_in_categorical_variable(_view, _feature)
                
                _feature_data_type = self._file_format_ref[_view][_feature]['data_type']
                
                if _feature_data_type == DataType.DATETIME.name:
                    self.check_datetime_variable_compliance(_view,
                                                           _feature)
                    
                if _feature_data_type == DataType.KEY.name:
                    self.check_key_variable_compliance(_view,
                                                      _feature)
    
    def check_number_of_samples(self, min_nb_samples: int, view_name: str='') -> bool:
        #Checking samples limit

        feature_name ='ALL' 
        sample_count = self._data_frame.shape[0]
        
        self._warning_logger.write_new_entry(PreProcessingChecks.N_SAMPLES_BELOW_THRESHOLD)
        if sample_count < min_nb_samples:
            success = False
            try:
                warning_msg = raise_warning(PreProcessingChecks.N_SAMPLES_BELOW_THRESHOLD,
                                            view_name, min_nb_samples, sample_count)
            except check_exception.MinimumSamplesViolatedException as err:
                print(err)
                self._warning_logger.add_exception(err)
                warning_msg = str(err)
            #message = critical_warning.display(f'Samples count exceeds the threshold limit {MIN_NB_SAMPLES}')
        else:
            success = True
            warning_msg = 'Test passed'

        self._warning_logger.write_checking_result(success,
                                                   warning_msg,
                                                   feature_name,
                                                   view_name)

        return success
    
    def check_feature_exists_in_dataset(self,
                                        view: str,
                                        feature_name: str) -> bool:
        renamed_feature_name = self._get_feature_defined_in_format_file(view, feature_name)
        if renamed_feature_name in self._data_frame.columns:
            success = True
        else:
            success = False
            self._warning_logger.write_new_entry(PreProcessingChecks.MISSING_FEATURE)
            try:
                raise_warning(PreProcessingChecks.MISSING_FEATURE,
                                        feature_name)
            except check_exception.MissingFeatureException as exc:
                warning_msg = str(exc)
                self._warning_logger.add_exception(exc)
            self._warning_logger.write_checking_result(success, warning_msg, feature_name)
            
        return success
    
    def check_missing_entry_format_file_ref(self,
                                            view:str,
                                            feature_name:str) -> bool:
        """Tests if format file ref is parsable"""
        
        success = True
        warning_msg = 'Test passed'
        
        renamed_feature_name = self._get_feature_defined_in_format_file(view, feature_name)
        
        _view_format_file = self._file_format_ref[view]
        _feature_format_file = _view_format_file.get(feature_name)
        if _feature_format_file is not None:
            _data_format_name = _feature_format_file.get('data_format')
            _data_type_name = _feature_format_file.get('data_type')
        else:
            _data_format_name, _data_type_name = None, None

        self._warning_logger.write_new_entry(PreProcessingChecks.INCORRECT_FORMAT_FILE)
        if _data_format_name is None :
            #success: bool, msg:str='', feature_name:str='
            warning_msg = raise_warning(PreProcessingChecks.INCORRECT_FORMAT_FILE,
                                    self._file_format_ref_name,
                                    renamed_feature_name)

            success = False
        self._warning_logger.write_checking_result(success,
                                             warning_msg,
                                             feature_name)  
        return success
    
    def check_correct_variable_sub_type(self, 
                                        view_name:str,
                                        feature_name:str,
                                        ) -> bool:
        """checks consistancy between general data type and subtype"""
        
        renamed_feature_name = self._get_feature_defined_in_format_file(view_name,
                                                                        feature_name)
        
        column = self._data_frame[renamed_feature_name]
        
        _feature_format_ref = self._file_format_ref[view_name][feature_name]
        success = True
        warning_msg = 'test passed'
        data_format_name = _feature_format_ref.get('data_format')
        data_type_name = _feature_format_ref.get('data_type')
        data_type_values = _feature_format_ref.get('values')
        data_type = None
        #feature_name = column.name

        # first test
        self._warning_logger.write_new_entry(PreProcessingChecks.DATA_TYPE_MISMATCH)
        if data_format_name is None or data_type_name is None:

            warning_msg = 'test skipped'
        else:
            try:
                data_type = utils.find_data_type(data_format_name, data_type_name)
                warning_msg = 'test passed'
            except ValueError as err:
                warning_msg = raise_warning(PreProcessingChecks.DATA_TYPE_MISMATCH, 
                                            data_format_name, data_type_name)
                success = False

        self._warning_logger.write_checking_result(success, warning_msg, feature_name)

        # second test 
        self._warning_logger.write_new_entry(PreProcessingChecks.INCORRECT_DATA_TYPE)
        if data_format_name is None or data_type_values is None:  # last condition is to check if data type is UNKNOWN
            warning_msg = 'test skipped'
        else:
            # first, we need to know if value is a dtetime or not
            # (basically, any strings or integers can be a datetime)
            if data_type_name == DataType.DATETIME.name:
                is_date = True
            else:
                is_date = False
            actual_dtype = utils.infer_type(column, is_date=is_date)

            if data_type is not None:  
                _have_correct_data_type = any(t == actual_dtype for t in data_type.value)
                if not _have_correct_data_type:
                    warning_msg = raise_warning(PreProcessingChecks.INCORRECT_DATA_TYPE, 
                                                feature_name,
                                                data_type_name, str(actual_dtype))
                    success = False
                else:
                    warning_msg = 'test passed'
                self._warning_logger.write_checking_result(success, warning_msg, feature_name)

        return success
    
    
    def check_missing_values(self, 
                             view_name: str,
                         feature_name: str,
                         threshold: int=None) -> bool:
        """checks if missing data are present in column, and triggers error depending
        of the fact that missing data are whether allowed or not in the format_ref_file"""
        
        if self._min_nb_samples is  None:
            success = None
        else:
            success = True
            if threshold is not None:
                threshold = self._min_nb_samples
            renamed_feature_name = self._get_feature_defined_in_format_file(view_name,
                                                                            feature_name)
            _column = self._data_frame[renamed_feature_name]
            _feature_format_ref = self._file_format_ref[view_name][feature_name]
            _is_missing_data = utils.check_missing_data(_column)
            _is_missing_values_authorized = _feature_format_ref.get('is_missing_values', 'test_skipped')
            

            
            self._warning_logger.write_new_entry(PreProcessingChecks.MISSING_DATA_ALLOWED)

            if _is_missing_values_authorized == 'test_skipped':
                warning_msg = 'Test skipped'
                success = None
            elif _is_missing_data:
                success = False
                # test fails: 
                if _is_missing_values_authorized:
                    # case where missing values are present BUT allowed
                    warning_msg = raise_warning(PreProcessingChecks.MISSING_DATA_ALLOWED,
                                            feature_name)
                else:
                    # case where missing values are present AND NOT allowed
                    try:
                        warning_msg = raise_warning(PreProcessingChecks.MISSING_DATA_NOT_ALLOWED,
                                                feature_name)
                    except check_exception.MissingDataException as err:
                        print(err)
                        self._warning_logger.add_exception(err)
                        warning_msg = str(err)
            else:
                # test passed
                warning_msg = 'Test passed'

            self._warning_logger.write_checking_result(success, warning_msg, feature_name)

        return success
    
    
    def check_lower_bound(self,
                          view_name:str,
                          feature_name:str) -> bool:
    
        _renamed_feature_name = self._get_feature_defined_in_format_file(view_name,
                                                                        feature_name)
        _column = self._data_frame[_renamed_feature_name]
        
        # remove nan (missing values) from vriable 
        _column_without_nan = _column.dropna()
        _feature_format_ref = self._file_format_ref[view_name][feature_name]
        lower_bound = _feature_format_ref.get('lower_bound')

        self._warning_logger.write_new_entry(PreProcessingChecks.OUTLIER_DETECTION_LOWER_BOUND)
        if lower_bound is not None:

            # should work for both numerical and datetime data types

            is_lower_bound_correct = np.all(_column_without_nan >= lower_bound)


            if not is_lower_bound_correct:
                warning_msg = raise_warning(PreProcessingChecks.OUTLIER_DETECTION_LOWER_BOUND,
                                            feature_name, lower_bound)
            else:
                warning_msg = 'Test passed'
        else:
            warning_msg = 'Test skipped'
            is_lower_bound_correct = None
        self._warning_logger.write_checking_result(is_lower_bound_correct, warning_msg, feature_name)

        return is_lower_bound_correct
    
    
    def check_upper_bound(self, view_name: str, feature_name: str) -> bool:
        _renamed_feature_name = self._get_feature_defined_in_format_file(view_name,
                                                                        feature_name)
        
        _column = self._data_frame[_renamed_feature_name]
        _feature_format_ref = self._file_format_ref[view_name][feature_name]
        
        # remove nan (missing values) from vriable 
        _column_without_nan = _column.dropna()
        upper_bound = _feature_format_ref.get('upper_bound')

        self._warning_logger.write_new_entry(PreProcessingChecks.OUTLIER_DETECTION_UPPER_BOUND)
        if upper_bound is not None:
             # should work for both numerical and datetime data sets
            is_upper_bound_correct = np.all(_column_without_nan <= upper_bound)

            if not is_upper_bound_correct:
                warning_msg = raise_warning(PreProcessingChecks.OUTLIER_DETECTION_LOWER_BOUND,
                                            feature_name, upper_bound)
            else:
                warning_msg = 'Test passed'

        else:
            warning_msg = 'Test skipped'
            is_upper_bound_correct = None

        self._warning_logger.write_checking_result(is_upper_bound_correct, warning_msg, feature_name)
        return is_upper_bound_correct
    
    
    def check_values_in_categorical_variable(self, 
                                             view_name:str,
                                             feature_name:str)-> bool:
        """Checks if values are contained in categorical variables"""

        _renamed_feature_name = self._get_feature_defined_in_format_file(view_name,
                                                                        feature_name)
        column = self._data_frame[_renamed_feature_name]
        
        _feature_format_ref = self._file_format_ref[view_name][feature_name]
        categorical_values = _feature_format_ref.get('categorical_values')

        self._warning_logger.write_new_entry(PreProcessingChecks.INCORRECT_VALUES_CATEGORICAL_DATA)
        if categorical_values is None:
            warning_msg = 'test skipped'
            success = None
        else:
            unique_values = utils.unique(column)
            success = True
            for val in unique_values:
                if val not in categorical_values and not utils.is_nan(val):
                    warning_msg = raise_warning(PreProcessingChecks.INCORRECT_VALUES_CATEGORICAL_DATA,
                                               feature_name, ", ".join(categorical_values), val)
                    success = False
            if success:
                warning_msg = 'test passed'
        self._warning_logger.write_checking_result(success, warning_msg, feature_name)
        return success

    def check_missing_values_threshold(self,
                                       view_name: str,
                                       feature_name: str,
                                       threshold: int = None) -> bool:
        #Checking if missing values exceed threshold limit(50%)
        
        if self._min_nb_missing_samples is None:
            success = None
        else:
            if threshold is not None:
                threshold = self._min_nb_missing_samples
            _renamed_feature_name = self._get_feature_defined_in_format_file(view_name,
                                                                            feature_name)
            _column = self._data_frame[_renamed_feature_name]
            _feature_format_ref = self._file_format_ref[view_name][feature_name]
            
            min_nb_missing_data = math.ceil((threshold/100)*_column.shape[0])

            self._warning_logger.write_new_entry(PreProcessingChecks.N_MISSING_DATA_ABOVE_THRESHOLD)
            n_missing_data = _column.isnull().sum()
            if (n_missing_data>min_nb_missing_data):
                success = False
                #message = critical_warning.display(f'Missing value exceeds threshold limit {MIN_NB_MISSING_DATA}',col) 
                warning_msg = raise_warning(PreProcessingChecks.N_MISSING_DATA_ABOVE_THRESHOLD,
                                            feature_name, n_missing_data,
                                            min_nb_missing_data)
            else:
                success = True
                warning_msg ='Test passed'

            #report['check_missing_values_limit'] = report_details
            self._warning_logger.write_checking_result(success, warning_msg, feature_name)
        return success
    
    def check_key_variable_compliance(self, 
                                      view_name:str,
                                      feature_name:str) -> bool:
        """Performs data sanity check over variable of type `KEY`
        warning should be Critical warnings
        """
        # variables initialisation

        _renamed_feature_name = self._get_feature_defined_in_format_file(view_name,
                                                                        feature_name)
        _column = self._data_frame[_renamed_feature_name]
        _feature_format_ref = self._file_format_ref[view_name][feature_name]
        
        # 1. check unicity of values in column

        n_unique_samples = utils.unique(_column, number=True)
        n_samples = _column.shape[0]

        

        self._warning_logger.write_new_entry(PreProcessingChecks.KEY_UNICITY_VIOLATED)
        if n_unique_samples != n_samples:
            success = False
            warning_msg = raise_warning(PreProcessingChecks.KEY_UNICITY_VIOLATED,
                                       feature_name,)
        else:
            warning_msg = 'Test passed'
            success = True
        self._warning_logger.write_checking_result(success, warning_msg, feature_name)

        return success

    def check_datetime_variable_compliance(self,
                                           view_name:str,
                                           feature_name:str) -> bool:
        """additional data sanity checks for datetime variable"""
        # test 1. check if datetime is parsable
        
        _renamed_feature_name = self._get_feature_defined_in_format_file(view_name,
                                                                        feature_name)
        _column = self._data_frame[_renamed_feature_name]
        _feature_format_ref = self._file_format_ref[view_name][feature_name]
        
        # remove missing values (nan) from column
        _column_without_nan = _column.dropna()
        are_datetime_parsables =  np.all(_column_without_nan.apply(utils.is_datetime))
        
        self._warning_logger.write_new_entry(PreProcessingChecks.INCORRECT_DATETIME_DATA)

        if not are_datetime_parsables:
            
            print('Warning: at least one variable is not a datetime')
            warning_msg = raise_warning(PreProcessingChecks.INCORRECT_DATETIME_DATA,
                                        feature_name)

        else:
            
            warning_msg = 'Test passed'
        self._warning_logger.write_checking_result(are_datetime_parsables,
                                                     warning_msg,
                                                     feature_name) 

        return are_datetime_parsables
    