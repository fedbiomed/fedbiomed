from enum import Enum
import os
from typing import List, Tuple, Dict, List, Any
import pprint

import pandas as pd

from demo_poc_data_wrapper.data_amputation_methods import ImputationMethods
from demo_poc_data_wrapper import utils
from demo_poc_data_wrapper.data_type import DataType

def get_yes_no_msg() -> str:
    msg_yes_or_no_question = '1) YES\n2) NO\n'   
    return msg_yes_or_no_question

def parse_yes_no_msg(resp: str) -> bool:
    """implements logic to parse yes or no msg"""
    yes_or_no_question_key = {'1': True,
                    '2': False}
    return yes_or_no_question_key.get(resp)

def get_data_type_selection_msg(available_data_type:List[Enum],
                               ign_msg: str = 'ignore this column') ->Tuple[str, int]:
    
    
    n_available_data_type = len(available_data_type)
    msg = ''

    
    for i, dtype in enumerate(available_data_type):
        msg += '%d) %s \n' %  (i+1, dtype.name)
    
    ignoring_key = i+2  # add ingoring entry
    msg += f'%d) %s\n' % (ignoring_key, ign_msg)
    
    return msg, ignoring_key

def ask_for_data_imputation_parameters(parameters_to_ask_for: List[str]) -> Dict[str, Any]:
    """asks for user parameters for data imputation"""
    params_retrieved_from_user = {}
    for param_name in parameters_to_ask_for:
        param = input(f"please specify {param_name} value:\n")
        params_retrieved_from_user[param_name] = param
        
    return params_retrieved_from_user

def get_data_imputation_methods_msg(d_type: type = None) -> Tuple[str, Dict[str, Enum]]:
    msg = 'Please select the following method for filling missing values (if some are found)\n'
    
    #available_methods = [method for ethod in ImputationMethods]
    select_action = {}
    i = 1
    
    for  imput_method in ImputationMethods:
        if d_type is not None:
            
            is_d_type_in_sub_type = utils.check_data_type_consistancy(imput_method.data_type, 
                                                               d_type)
            if not is_d_type_in_sub_type:
                continue # data type doesnot match method amputation requirments
        msg += '%d) %s\n' % (i, imput_method.name)
        select_action[str(i)] = imput_method
        i += 1

    # ignore key
    msg += '%d) No method\n' % i 
    select_action[str(i)] = None
    return msg, select_action
    
def no_methods(*kwargs):
    return None

    
    
def get_from_user_dataframe_format_file(dataset: pd.DataFrame) -> Dict[str, Any]:
    ##
    # variable initialisation
    data_format_file = {}
    
    dataset_columns = dataset.columns
    dataset_columns_length = len(dataset_columns)
    
    
    available_data_type = [d_type for d_type in DataType]  # get all available data types
    
    for n_feature, feature in enumerate(dataset_columns):
        print(f'displaying first 10 values of feature {feature} (n_feature: {n_feature+1}/{dataset_columns_length})') 
        #_file_name = os.path.basename(tabular_data_file)
        #data_format_files[_file_name] = data_format_file
        #print(tabulate(dataset[feature].head(10).values()))
        pprint.pprint(dataset[feature].head(10))  # print first 10 lines of feature value
        print(f'number of differents samples: {utils.unique(dataset[feature], number=True)} / total of samples: {dataset[feature].shape[0]}')
        
        msg_data_type_selection, ignoring_id = get_data_type_selection_msg(available_data_type)
        msg_data_type_selection = f'specify data type for {feature}:\n' + msg_data_type_selection
        
        # ask user about data type
        data_format_id = get_user_input(msg_data_type_selection,
                                       
                                       n_answers=ignoring_id)
        
        if int(data_format_id) > ignoring_id - 1:
            # case where user decide to ingore column: go to next iteration (next feature)
            print(f"Ignoring feature {feature}")
            continue
        else:
            # case where user selected a data type: add data type and info to the format file
            data_format = available_data_type[int(data_format_id)-1]
            d_type = dataset[feature].dtype  
            # TODO: rename data_type into d_type for consistancy sake
            n_data_type, types = utils.get_data_type(data_format, d_type)
            
        # KEY and DATETIME type 
        if data_format is DataType.KEY or data_format is DataType.DATETIME:  
            # for these data type, missing values are disabled by default
            is_missing_values_allowed = False
            amputation_method = None
            amputation_method_parameters = None
        else: 
            # ask user if missing values are allowed for this specific variable
            ## message definition
            msg_yes_or_no_question = get_yes_no_msg()
            msg_data_imputation_methods, data_imputation_methods = get_data_imputation_methods_msg(d_type=d_type)
            n_data_imputation_method = len(data_imputation_methods)
            msg_yes_or_no_question = f'Allow {feature} to have missing values:\n' + msg_yes_or_no_question
            
            missing_values_user_selection = get_user_input(msg_yes_or_no_question,
                                                        n_answers=2)
            is_missing_values_allowed = parse_yes_no_msg(missing_values_user_selection)
            
            amputation_method = None
            amputation_method_parameters = None
            if is_missing_values_allowed:
                # let user select amputation method if missing data are allowed
                amputation_method_user_selection = get_user_input(msg_data_imputation_methods,
                                                                n_answers=n_data_imputation_method)
                
                amputation_method_selected = data_imputation_methods.get(amputation_method_user_selection)
                
                if amputation_method_selected is not None:
                    amputation_method = amputation_method_selected.name
                    if amputation_method_selected.parameters_to_ask_user is not None:
                        print(f'Selected: {amputation_method}\n')
                        amputation_method_parameters = ask_for_data_imputation_parameters(amputation_method_selected.parameters_to_ask_user)
                        print('amput param', amputation_method_parameters)
            
                
        data_format_file[feature] = {'data_format': data_format.name,
                                     'data_type': n_data_type.name,
                                     'values': str(d_type),
                                     'is_missing_values': is_missing_values_allowed,
                                     'data_amputation_method': amputation_method,
                                     'data_amputation_parameters': amputation_method_parameters
                                    }
        
    return data_format_file
            
def get_user_input(msg:str,  n_answers:int) -> str:
    """"""
    is_column_parsed = False
    while not is_column_parsed:
        #data_format_id = input(f'specify data type for {feature}:\n' + msg )
        resp = input(msg)
        if resp.isdigit() and int(resp) <= n_answers and int(resp)>0:
            # check if value passed by user is correct (if it is integer,
            # and whithin range [1, n_available_data_type])
            is_column_parsed = True

        else:
            print(f'error ! {resp} value not understood')
            
    return resp

### CLI to use when dataset is available


def get_from_user_multi_view_dataset_fromat_file(datasets: Dict[str, pd.DataFrame])-> Dict[str, pd.DataFrame]:
    
    data_format_files = {}
    
    for tabular_data_file in datasets.keys():
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print(f"+++++++ Now parsing view: {tabular_data_file} +++++++")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        data_format_file = get_from_user_dataframe_format_file(datasets[tabular_data_file])
        if data_format_file:
            # (above condition avoids adding empty views)
            _file_name = os.path.basename(tabular_data_file)
            data_format_files[_file_name] = data_format_file
        
    return data_format_files
