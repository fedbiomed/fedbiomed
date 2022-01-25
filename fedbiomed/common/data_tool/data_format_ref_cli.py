from enum import Enum
import os
from typing import List, Tuple, Dict, List, Any, Callable, Iterator
import pprint
import copy
import pandas as pd

from fedbiomed.common.data_tool.data_imputation_methods import ImputationMethods
from fedbiomed.common.data_tool import utils
from fedbiomed.common.data_tool.data_type import DataType, DataTypeProperties


GLOBAL_THRESHOLDS = 'global_thresholds'  # additional entry used for specifying global parameters (eg nb )


def get_yes_no_msg() -> str:
    msg_yes_or_no_question = '1) YES\n2) NO\n'   
    return msg_yes_or_no_question


def parse_yes_no_msg(resp: str) -> bool:
    """implements logic to parse yes or no msg"""
    yes_or_no_question_key = {'1': True,
                              '2': False}
    return yes_or_no_question_key.get(resp)


def get_data_type_selection_msg(available_data_type: List[Enum],
                                ign_msg: str = 'ignore this column') -> Tuple[str, int]:
    
    #n_available_data_type = len(available_data_type)
    msg = ''
    for i, dtype in enumerate(available_data_type):
        msg += '%d) %s \n' % (i + 1, dtype.name)
    
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

def select_unselect(selected_items: List[str], item:str, verbose:bool=False):
    """Appends or removes an item from a list depending whether it
    already belongs to the list or not

    Args:
        selected_items (List[str]): [description]
        item (str): [description]
    """
    if item in selected_items:
        selected_items.remove(item)
        if verbose:
            print(f'removed {item}')
    else:
        selected_items.append(item)
        if verbose:
            print(f'added {item}')

def _check_if_features_selected(selected_views_features: Dict[str, List[str]]) -> bool:
    is_one_feature_selected = False
    
    for view in selected_views_features.keys():
        if selected_views_features[view]:
            # check whether the list contained in dictionary is empty or not
            is_one_feature_selected = True
            break
    return is_one_feature_selected
    
def ask_select_variable_msg(view_feature_names: Dict[str, List[str]], 
                            selected_views_idx: List[str],
                            selected_views_features: Dict[str, List[str]],
                            imput_view: str,
                            imput_feature: str) -> str:
    """ for data imputation methods"""
    _is_features_selected = False
    #_is_one_feature_selected = False
    view_names = list(view_feature_names.keys())
    
    while not _is_features_selected:
        print('get msg', selected_views_idx)
        
        view_selection_msg = ['select all views']
        _is_one_feature_selected = _check_if_features_selected(selected_views_features)
        if _is_one_feature_selected:
            view_selection_msg.append('Finish feature selection')
        view_msg, last_key = get_select_msg(view_names,
                                            'Please select view that contained desired feature:',
                                            inter_msg='(selected)',
                                            inter_msg_idx=selected_views_idx,
                                            final_msgs=view_selection_msg
                                            )
        _answer = int(get_user_input(view_msg, last_key))
    
        
        selected_view, selected_feature = None, None
        
        # define condition when user hits 'select all key', depending whether features
        # have been added or not
        _select_all_view_cond1 = _answer == last_key - 1 and _is_one_feature_selected
        _select_all_view_cond2 = _answer == last_key and not _is_one_feature_selected
        _select_all_view_cond = _select_all_view_cond1 or _select_all_view_cond2
        if _answer == last_key and _is_one_feature_selected:
            # confirm choice
                _is_features_selected = True
                continue
            
        elif _select_all_view_cond:
            # user entered 'select all views'
            selected_view = view_names
            selected_views_idx.extend(view_names)
            selected_views_idx = list(set(selected_views_idx))
            for view in view_names:
                selected_views_features[view] = copy.deepcopy(view_feature_names[view])
        else:
            selected_view = view_names[_answer - 1]
            
            #selected_views_idx.append(_answer)
            print('Please select a feature in %s' % selected_view)
            if imput_view == selected_view and imput_feature in view_feature_names[selected_view]:
                # remove feature that is imputed from the name list
                
                view_feature_names[selected_view].remove(imput_feature)
            feature_names = copy.deepcopy(view_feature_names[selected_view])
            selected_feature_names = selected_views_features.get(selected_view)
            
            feature_msg, last_key = get_select_msg(feature_names,
                                                    f'Please select feature from view : {selected_view}',
                                                    inter_msg='(selected)',
                                                    inter_msg_idx=selected_feature_names,
                                                    final_msgs=['select all features',
                                                                'return to view'])
            _answer = int(get_user_input(feature_msg, last_key))

            if _answer == last_key:
                _is_features_selected = True
            elif _answer == last_key - 1:
                # select all features whithin the view
                selected_views_features[selected_view] = feature_names
                
            else:
                selected_feature = feature_names[_answer - 1]

                select_unselect(selected_views_features[selected_view],
                                selected_feature, verbose=True) 
                
            if sorted(selected_feature_names) == sorted(feature_names):
                # case where all feature in a view has been selected
                selected_views_idx.append(selected_view)
            elif selected_view in selected_views_idx:
                selected_views_idx.remove(selected_view)

        print('end', selected_views_idx, _is_features_selected, _is_one_feature_selected)

def get_select_msg(iterator: Iterator[str], 
                   begining_msg: str='',
                   inter_msg: str='',
                   inter_msg_idx: List[str] = None,
                   final_msgs: List[str] = None,
                   method:str = None) -> Tuple[str, int]:   
    
    msg = begining_msg + '\n'
    print('get msg', inter_msg, inter_msg_idx)
    for i, line in enumerate(iterator):
        
        if method is not None:
            line = getattr(line, method)
        if inter_msg_idx is not None:
            if line in  inter_msg_idx:
                _inter_msg = inter_msg
            else:
                _inter_msg = ''
        else:
            _inter_msg = ''
        msg += '%d) %s %s\n' % (i + 1, line, _inter_msg)
    last_key = i + 1
    if final_msgs is not None:
        for final_msg in final_msgs:
            last_key += 1
            msg += '%d) %s\n' % (last_key, final_msg)
    return msg, last_key


def ask_user_threshold( 
                       intro_msg: str = "", 
                       msg: str = "") -> int:
    threshold = None

    intro_msg += get_yes_no_msg()
    # first ask user if he wants to add a threshold of minimal number of samples
    answer = get_user_input(intro_msg, 2)
    _does_user_want_threshold = parse_yes_no_msg(answer)
    
    if _does_user_want_threshold:
        # case where user wants miimal number of samples of threshold
        _is_entered_value_correct = False
        while not _is_entered_value_correct:
            threshold = input(f'enter threshold {msg}\n')
            if threshold.isdecimal() and int(threshold) > 0:
                threshold = int(threshold) 
                # check if entered value is correct (it must be an integer)
                _is_entered_value_correct = True
            else:
                print(f'Value {threshold} not a a valid Threshold! please retry')
        threshold = int(threshold)
    return threshold


def ask_minimum_nb_samples() -> int:
    msg_add_threshold = "Do you want to add a threshold for the minimum number of samples each dataset should contain?\n"
    msg = "minimal number of samples per dataset"
    threshold = ask_user_threshold(msg_add_threshold,
                                   msg)
    
    return threshold

def ask_minimum_of_missing_data() -> int:
    msg_add_threshold = "Do you want to add a threshold for the minimum number of missing data each feature should contain?\n"
    # first ask user if he wants to add a threshold of minimal number of samples
    msg = "minimal number of missing sample per variable"
    
    threshold = ask_user_threshold(msg_add_threshold,
                                   msg)
    
    return threshold
  
def ask_for_variable_to_apply_data_imputation(view_feature_names: Dict[str, List[str]],
                                              view: str,
                                              feature: str) -> Dict[str, List[str]]:
    """
    Asks on which variable to apply data imputation method
    """
    
    selected_view_idx = []
    selected_views_features = {view : [] for view in view_feature_names.keys()}
    
    _is_finished = False
    
    resume_msg = "Do you want to add more variable to apply to data imputation method ?\n"
    resume_msg += get_yes_no_msg()
    while not _is_finished:
        ask_select_variable_msg(view_feature_names,
                                selected_view_idx,
                                selected_views_features,
                                view, feature)
        answer = get_user_input(resume_msg, 2)
        answer = parse_yes_no_msg(answer)
        _is_finished = not answer
    
    # TODO: what to do if user doesnot want to add a variable? throw an error?
    
    return selected_views_features

def get_data_imputation_methods_msg(d_type: type = None) -> Tuple[str, Dict[str, Enum]]:
    msg = 'Please select the following method for filling missing values (if some are found)\n'
    
    #available_methods = [method for ethod in ImputationMethods]
    avail_action = {}
    i = 1
    
    for imput_method in ImputationMethods:
        if d_type is not None:
            
            is_d_type_in_sub_type = utils.check_data_type_consistancy(imput_method.data_type, 
                                                                      d_type)
            if not is_d_type_in_sub_type:
                continue # data type doesnot match method imputation requirements
        msg += '%d) %s\n' % (i, imput_method.name)
        avail_action[str(i)] = imput_method
        i += 1

    # ignore key
    msg += '%d) No method\n' % i  # specify in data format file
    # to not use specific method
    avail_action[str(i)] = None
    return msg, avail_action
    
def no_methods(*kwargs):
    return None

def get_from_user_dataframe_format_file(dataset: pd.DataFrame,
                                        view: str,
                                        view_feature_names: Dict[str, List[str]] = None) -> Dict[str, Any]:
    ##
    # variable initialisation
    data_format_file = {}
    
    dataset_columns = dataset.columns
    dataset_columns_length = len(dataset_columns)
    
    available_data_type = [d_type for d_type in DataType]  # get all available data types
    
    for n_feature, feature in enumerate(dataset_columns):  # iterates over features
        print(f'displaying first 10 values of feature {feature} in view: {view} (n_feature: {n_feature+1}/{dataset_columns_length})') 

        pprint.pprint(dataset[feature].head(10))  # print first 10 lines of feature values
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
            # get property of data_type
            data_type_property = utils.get_data_type_propreties(data_format)
            
            if data_type_property.date_ambiguity:
                is_date = ask_if_datetime_variable(feature)
            elif data_type_property.date_format:
                is_date = True  # if data type has a date format, then it is a datetime variable
            else:
                is_date = False  # should not occur (property error)
            d_type = utils.infer_type(dataset[feature], data_format, is_date)  
            
            # TODO: rename data_type into d_type for consistancy sake
            _n_data_type, types = utils.get_data_type(data_format, d_type)
            
        # check if data type  doesnot allow missing values (such as KEY and DATETIME types) 
        if not data_type_property.allow_missing_values:  
            # for these data type, missing values are disabled by default
            is_missing_values_allowed = False
            data_imputation_params = {}
        else: 
            # ask user if missing values are allowed for this specific variable
            ## buliding message definition
            msg_yes_or_no_question = get_yes_no_msg()
            msg_yes_or_no_question = f'Allow {feature} to have missing values:\n' + msg_yes_or_no_question
            missing_values_user_selection = get_user_input(msg_yes_or_no_question,
                                                           n_answers=2)
            is_missing_values_allowed = parse_yes_no_msg(missing_values_user_selection)
            
            # selected_variables = None
            data_imputation_params = {}
            if is_missing_values_allowed:
                # let user select imputation method if missing data are allowed
                data_imputation_params = ask_for_data_imputation_method(view_feature_names,
                                                                        d_type,
                                                                        view, feature)
        
        if isinstance(types, list):
            # convert type to string so they can be saved into JSON
            types = [str(x) for x in types]  
            

        data_format_file[feature] = {'data_format': data_format.name,
                                     'data_type': _n_data_type.name if _n_data_type is not None else None,
                                     'values': types,
                                     'is_missing_values': is_missing_values_allowed,
                                     'data_imputation_method': data_imputation_params.get('data_imputation_method'),
                                     'data_imputation_parameters': data_imputation_params.get('data_imputation_parameters'),
                                     'data_imputation_variables': data_imputation_params.get('data_imputation_variables')
                                     }
        
    return data_format_file
            
def get_user_input(msg:str,  n_answers:int) -> str:
    """
    Asks user a question contained into msg. User is
    supposed to enter an integer as a multiple choice question.
    Eg: how are you?:
    1) great
    2) good
    3) could be worst
    4) bad
    Asks the question again if user's input is not an integer
    and if this integer is not whitin range [1, <n_answers>]
    
    Args:
        msg (str): [description]
        n_answers (int): [description]

    Returns:
        str: User's answer
    """
    is_column_parsed = False
    while not is_column_parsed:

        resp = input(msg)
        if resp.isdigit() and int(resp) <= n_answers and int(resp)>0:
            # check if value passed by user is correct (if it is integer,
            # and whithin range [1, n_available_data_type])
            is_column_parsed = True

        else:
            print(f'error ! {resp} value not understood')
            
    return resp

### CLI to use when dataset is available

def ask_if_datetime_variable(variable_name:str=''):
    msg = f"Datetime ambiguity: Please specify if variable {variable_name} is a date or not\n"
    msg += get_yes_no_msg()
    answer = get_user_input(msg, 2)
    is_datetime = parse_yes_no_msg(answer)
    return is_datetime
    
def get_from_user_multi_view_dataset_fromat_file(datasets: Dict[str, pd.DataFrame])-> Dict[str, pd.DataFrame]:
    
    data_format_files = {}
    
    # first ask user for minimum number of samples + maximum number of missing data thresholds
    print('+++++++++ Editing Global Thresholds +++++++++++')
    _min_number_samples = ask_minimum_nb_samples()
    _min_number_missing_data = ask_minimum_of_missing_data()
    data_format_files.update({GLOBAL_THRESHOLDS: 
                                    {'min_nb_samples': _min_number_samples,
                                     'min_nb_missing_samples': _min_number_missing_data}})
    
    view_feature_name = get_view_feature_name_from_dataframe(datasets)
    for tabular_data_file in datasets.keys():
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print(f"+++++++ Now parsing view: {tabular_data_file} +++++++")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        data_format_file = get_from_user_dataframe_format_file(datasets[tabular_data_file],
                                                               tabular_data_file,
                                                               view_feature_name)
        if data_format_file:
            # (above condition avoids adding empty views)
            _file_name = os.path.basename(tabular_data_file)
            data_format_files[_file_name] = data_format_file
        
    return data_format_files

def get_view_feature_name_from_format_file(format_file_ref: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    return {v: [f for f in format_file_ref[v].keys()] for v in utils.get_view_names(format_file_ref)}

def get_view_feature_name_from_dataframe(data_frame: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
    return {v : list(df.columns) for v, df in data_frame.items()}
    
    
# edition of format_file_ref

def edit_format_file_ref(format_file_ref: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    
    # CLI for editing `format_file_ref`, a file containing information about each variable
    # in a tabular dataset
    print(f'+++++++ Now editing format file ref ++++++++++')
    
    ## variables initialization

    _file_names = utils.get_view_names(format_file_ref)
    # removes GLOBAL_TRESHOLDS key from list of keys
    # (because it is not a view but general parameters)
        
    _n_tot_files = len(_file_names)
    
    ## messages definition
    _messages = {
        'yes_or_no': get_yes_no_msg(),
        'edit': 'Which field should be modified?\n',
    }
    view_feature_names = get_view_feature_name_from_format_file(format_file_ref)
    
    # iterate over name of files (ie views)
    for i_file in range(_n_tot_files):
        # ask for each file if user wants to edt it
        _answer = get_user_input(f"Edit file: {_file_names[i_file]}?\n" + _messages['yes_or_no'], 2)
        _answer = parse_yes_no_msg(_answer)
        
        if _answer:
            # case where user wants to modify current view scheme
            _file_content = format_file_ref[_file_names[i_file]]  # get file (ie view) content
            
            ## variables initialization for parsing current view
            _features_names = list(_file_content.keys())
            _n_tot_feature = len(_features_names)
            
            # iterate over features found in view
            for i_feature in range(_n_tot_feature):
                feature_name = _features_names[i_feature]
                feature_content = _file_content[feature_name]
                feature_content = edit_feature_format_file_ref(feature_content,
                                                               feature_name,
                                                               view_feature_names,
                                                               _messages)
            format_file_ref[_file_names[i_file]].update({feature_name: feature_content})
            
    return format_file_ref

def create_msg_action_selection(data_type_propreties: Enum) -> Tuple[str, int, Dict[int, Callable], List[Any]]:
    # create edit selection message for user given data type
    # number of possible actions that depend of data type properties
    msg = ""
    action_counter = 1
    actions = {}
    
    # data type change command
    msg += "%d) data_type\n" % action_counter
    actions[str(action_counter)] = ask_for_data_type
    action_counter += 1
    
    if data_type_propreties.lower_bound:
        # lower bound edit command
        msg += "%d) lower bound\n" % action_counter
        actions[str(action_counter)] = ask_for_lower_bound
        action_counter += 1
        params_action = None
    
    if data_type_propreties.upper_bound:
        # upper bound edit command
        msg += "%d)upper bound\n" % action_counter
        actions[str(action_counter)] = ask_for_upper_bound
        action_counter += 1
        params_action = None
        
    if data_type_propreties.set_of_values:
        # value taken edit command
        msg += "%d) Values taken\n" % action_counter
        actions[str(action_counter)] = ask_for_categorical_values
        action_counter += 1
        params_action = None
        
    if data_type_propreties.date_format:
        # date formatter edit command
        msg += "%d) Date format\n" % action_counter
        actions[str(action_counter)] = ask_for_date_format
        action_counter += 1
        params_action = None
        
    if data_type_propreties.allow_missing_values:
        # change data method for data imputation
        msg += "%d) Data Value imputation method\n" % action_counter
        actions[str(action_counter)] = ask_for_data_imputation_method
        action_counter += 1
        params_action = ['view_feature_names']
        
    msg += "%d) Cancel Operation\n" % action_counter
    actions[str(action_counter)] = cancel_operation
    return msg, action_counter, actions, params_action

def select_action(
                  action: str,
                  possible_actions: Dict[str, Callable],
                  *params: List[Any]
                  ) -> Tuple[Dict[str, Any], bool]:
    
    # variable initialization
    is_cancelled = False
    new_field = None
    print('action', action, str(len(possible_actions.keys())),possible_actions)
    if action == str(len(possible_actions.keys())):
        is_cancelled = True
        
    else:
        # define action among the pool of possible actions
        _action_func = possible_actions[action]
        new_field = _action_func(*params)
    
    return new_field, is_cancelled

def cancel_operation():
    print("operation cancelled")
        
def ask_for_lower_bound(*kwargs) -> Dict[str, float]:
    _is_entered_value_correct = False
    while not _is_entered_value_correct:
        lower_bound = input('enter lower bound')
        if utils.isfloat(lower_bound) or utils.is_datetime(lower_bound):
            # check if entered value is correct (is a numerical value)
            _is_entered_value_correct = True
        else:
            print('Value not a Number! please retry')
    return {'lower_bound': float(lower_bound)}

def ask_for_upper_bound(*kwargs) -> Dict[str, float]:
    
    _is_entered_value_correct = False
    while not _is_entered_value_correct:
        upper_bound = input('enter upper bound')
        if utils.isfloat(upper_bound) or utils.is_datetime(upper_bound):
            # check if entered value is correct (is a numerical value)
            _is_entered_value_correct = True
        else:
            print('Value not a Number! please retry')
    return {'upper_bound': float(upper_bound)}


def _ask_for_data_type(data_type: Enum) -> Enum:
    """asks user for datatype contains in `data_type`
    If user selects `cancel`, it will return None
    """
    
    _available_data_type = [t for t in data_type]  # get all keys contain in data_type
    
    msg, _n_answer = get_data_type_selection_msg(data_type, ign_msg="cancel operation")
    data_type_selection = get_user_input(msg, _n_answer)
    
    if str(_n_answer) != data_type_selection:
        return _available_data_type[int(data_type_selection) - 1]
    
    
def ask_for_data_type(*kwargs) -> Dict[str, Any]:

    updates = None
    
    new_data_format = _ask_for_data_type(DataType)
    
    if new_data_format is not None: 
        # case where 'cancel operation' hasnot been selected
                
        updates = {'data_format': new_data_format.name}
        if  isinstance(next(iter(new_data_format.value)), Enum):
            # if subtypes are available
            new_data_type= _ask_for_data_type(new_data_format.value)
            new_values = list(map(lambda x: str(x), new_data_type.value))
            updates.update({'data_type': new_data_type.name, 'values': new_values})
        else:
            new_values = list(map(lambda x: str(x), new_data_format.value))
            updates.update({'values': new_values})
    return updates


def ask_for_data_imputation_method(view_feature_names: Dict[str, List[str]]=None,
                                   d_type: type=None,
                                   view:str=None,
                                   feature:str=None) -> Dict[str, Any]:
    
    # variables initialisation
    _imputation_method = None
    _imputation_method_parameters = None
    _imputatation_variables = None
    
    # get user message + dictionary mapping user responses to data imputation methods
    _msg_data_imputation_methods, _data_imputation_methods = get_data_imputation_methods_msg(d_type)
    print(_msg_data_imputation_methods, 'd_type', d_type)
    # ask for user selection
    _imputation_method_user_selection = get_user_input(_msg_data_imputation_methods,
                                                      n_answers=len(_data_imputation_methods))
    
    # select user data imputation method given user command
    _imputation_method_selected = _data_imputation_methods.get(_imputation_method_user_selection)

    if _imputation_method_selected is not None:
        _imputation_method = _imputation_method_selected.name
        
        if _imputation_method_selected.parameters_to_ask_user is not None:
            print(f'Method imputation selected: {_imputation_method}\n')
            _imputation_method_parameters = ask_for_data_imputation_parameters(_imputation_method_selected.parameters_to_ask_user)
            print('imput param', _imputation_method_parameters)
        if _imputation_method_selected.ask_variable_to_perform_imputation:
            _imputatation_variables = ask_for_variable_to_apply_data_imputation(view_feature_names,
                                                                                view, feature)
    updates = {'data_imputation_method': _imputation_method,
               'data_imputation_parameters': _imputation_method_parameters,
               'data_imputation_variables': _imputatation_variables}
    return updates

def ask_for_categorical_values(*kwargs) -> Dict[str, Any]:
    possible_values = input('enter possible values (separated by ",")')
    possible_values = possible_values.split(",")  # separate values passed by user into a list
    return {'categorical_values': possible_values}


def ask_for_date_format(*kwargs) -> Dict[str, Any]:
    # TODO : ask for date format (UTC, ....)
    msg = 'please enter date format:\n1)timetsamp\n2)ISO date format (YYYY-MM-DD)\n3)custom date format\n'
    user_selection = input(msg)
    # default date format
    msg = {'1': 'timestamp',
          '2': '(American default date format) mm/dd/yy',
          '3': '(Europeen default date format) dd/mm/yy',
           '4': 'ISO date format (YYYY-MM-DD)',
          '5': 'custom date format',
          '6': 'select timezone'}
    pass  # unfinished (posteponed)

def edit_feature_format_file_ref(feature_content: Dict[str, Any],
                                  feature_name: str,
                                  view_feature_name: Dict[str, Dict[str, Any]],
                                  messages: Dict[str, str]) -> Dict[str, Any]:
    """Edits a specific feature that belongs to a specific view within a format file"""
    

    _is_feature_unparsed = True  
    _is_cancelled = False  # whether parsing of current column has been cancelled or not
    _is_first_edit = True
    
    # iterate over number of feature contained in view, and ask for each feature if changes are needed
    while _is_feature_unparsed:
        data_format = feature_content.get('data_format')
        if _is_cancelled or not _is_first_edit:
            _f_answer = True
        else:
            _f_answer = get_user_input(f"Edit variable: {feature_name}? (type: {data_format})\n" + messages['yes_or_no'], 2)
            # ask if user wants to edit feature variables
            _f_answer = parse_yes_no_msg(_f_answer)
            _is_first_edit = False
        if _f_answer:
            # case where user wants to edit the current feature
            
            _msg = messages['edit']
            
            for data_type_properties in DataTypeProperties:
                if data_format == data_type_properties.name:
                    # get data property from data_format
                    select_msg, n_actions, possible_actions, params_action = create_msg_action_selection(data_type_properties)
                    _msg += select_msg
                    _edit_selection = get_user_input(_msg, n_actions)

                    if params_action is not None and 'view_feature_names' in params_action:
                        params_action = view_feature_name
                    _edited_field, _is_cancelled = select_action(
                                                                  _edit_selection,
                                                                  possible_actions,
                                                                  view_feature_name
                                                                )

            if not _is_cancelled:
                # if user has not cancelled field edition
                if _edited_field is not None:
                    feature_content.update(_edited_field)
             
                _c_answer = get_user_input(f"Continue Editing variable: {feature_name}?\n" + messages['yes_or_no'], 2)
                _is_feature_unparsed = parse_yes_no_msg(_c_answer)
            else:
                _is_feature_unparsed = False
                
        else:
            _is_feature_unparsed = False
            
    return feature_content