from enum import Enum
import csv
import json

from typing import List, Tuple, Union, Dict, Any, Iterator
import os

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
import dateutil
from dateutil.parser._parser import ParserError

from fedbiomed.common.data_tool.data_type import DataType, DataTypeProperties
from fedbiomed.common.data_tool.data_format_ref_cli import GLOBAL_THRESHOLDS

STR_MISSING_VALUE = ['nan']

class ExcelSignatures(Enum):
    XLSX = (b'\x50\x4B\x05\x06', 2, -22, 4)
    LSX1 = (b'\x09\x08\x10\x00\x00\x06\x05\x00', 0, 512, 8)
    LSX2 = (b'\x09\x08\x10\x00\x00\x06\x05\x00', 0, 1536, 8)
    LSX3 = (b'\x09\x08\x10\x00\x00\x06\x05\x00', 0, 2048, 8)
    
    def __init__(self, sig, whence, offset, size):
        self._sig = sig
        self._whence = whence
        self._offset = offset
        self._size = size

    @property 
    def signature(self) -> bytes:
        return self._sig
    
    @property
    def whence(self) -> int:
        return self._whence
    
    @property
    def offset(self) -> int:
        return self._offset
    
    @property
    def size(self) -> int:
        return self._size



def load_tabular_datasets(path:str) -> Dict[str, pd.DataFrame]:
    tabular_datasets = {}

    if os.path.isdir(path):
        print('directory found')
        _is_folder = True
        
        _tabular_data_files = os.listdir(path)
    else:
        print('file found')
        _is_folder = False
        _tabular_data_files = (path,)
        
    for tabular_data_file in _tabular_data_files:
        if _is_folder:
            tabular_data_file = os.path.join(path, tabular_data_file)
        
        _is_excel = excel_sniffer(tabular_data_file)
        _csv_delimiter, _csv_header = csv_sniffer(tabular_data_file)
        _view_name = os.path.basename(tabular_data_file)
        if _is_excel:
            tabular_datasets[_view_name] = load_excel_file(tabular_data_file)
        elif _csv_delimiter is not None:
            tabular_datasets[_view_name] = load_csv_file(tabular_data_file,
                                                               _csv_delimiter, 
                                                               _csv_header)
        else:
            print(f'warning: cannot parse {tabular_data_file}: not a tabular data file')
        
    return tabular_datasets

def load_csv_file(path:str, delimiter:str, header:int) -> pd.DataFrame:
    try:
        dataframe = pd.read_csv(path, delimiter=delimiter, header=header)
    except csv.Error as err:
        print('err', err, 'in file', path)
            
    return dataframe

#https://stackoverflow.com/questions/23515791/how-to-check-the-uploaded-file-is-csv-or-xls-in-python/23515973




def load_excel_file(path:str, sheet_name: Union[str, int]=0) -> pd.DataFrame:
    """May rely on openpyxl package"""
    #with open(path, 'r') as excl:
    #    _c = csv.DictReader(excl, dialect=csv.excel_tab)
    #    _delimiter = _c.dialect.delimiter
    
    dataframe = pd.read_excel(path, sheet_name=sheet_name)
    return dataframe


def excel_sniffer(path: str) -> bool:
    
    for excel_sig in ExcelSignatures:
        with open(path, 'rb') as f:
            f.seek(excel_sig.offset, excel_sig.whence)
            bytes = f.read(excel_sig.size)

            if bytes == excel_sig.signature:
                return True
            else:
                return False
            

def csv_sniffer(path:str) :
        
    with open(path, 'r') as csvfile:
        try:
            # do some operation on file using sniffer to make sure considered file
            # is a CSV file
            dialect = csv.Sniffer().sniff(csvfile.readline())
            delimiter = dialect.delimiter
            dialect.lineterminator
            has_header = csv.Sniffer().has_header(csvfile.readline())
            if has_header:
                header = 0
            else:
                header = None
        except (csv.Error, UnicodeDecodeError) as err:
            delimiter, header = None, None
            print('err', err, 'in file', path)
    return delimiter, header

def get_data_type(
                  d_format: Enum,
                  d_type: type) -> Tuple[Enum, List[Union[type, str]]]:
    # varibales initialisation
    present_d_types = []
    sub_d_type_format = d_format

    for avail_data_type in DataType:
        if d_format is avail_data_type:
            sub_dtypes = avail_data_type.value

            if not isinstance(sub_dtypes, str):
                # check if dtype has subtypes
                #(eg if datatype is QUANTITATIVE, subtype will be CONTINOUS or DISCRETE)
                if isinstance(next(iter(sub_dtypes)), Enum):
                    
                    for sub_dtype in sub_dtypes:
                        cond = False
                        for t in tuple(sub_dtype.value):
                            if d_type == t:
                                print('found ', t)
                        if any(d_type == t for t in tuple(sub_dtype.value)):
                            present_d_types.append(d_type)
                            sub_d_type_format = sub_dtype
                            print(d_type, t, present_d_types)
                            
                else:
                    # case where datatype doesnot have subtypes, eg DATETIME
                    if any(d_type == t for t in sub_dtypes):
                        present_d_types.append(d_type)
                    sub_d_type_format = d_format
            else:
                # case where d_format is a string of character, eg UNKNOWN
                sub_d_type_format = d_format
                present_d_types = None
    if isinstance(present_d_types, list) and not present_d_types:  # check if list is empty
        print(f"Warning: {d_type} type doesnot belong to data type {d_format.name}")
        present_d_types.append(d_type)
        sub_d_type_format = None
    return sub_d_type_format, present_d_types


def find_data_type(data_format_name: str, data_type_name: str=None) -> Enum:
    """Retrieves from a given data_format and data_type,
    the corresponding Enum class describing data
    
    Returns:
    Data_type (Enum): The data type class corresponding to the ones given
    in strings
    
    Raises:
    ValueError: case where arguments data_format_name and data_type_name 
    could not be recognized
    """
    
    ## varible initialisation
    data_type = None
    _is_data_format_unrecognized = True
    _is_data_type_unrecognized = True
    
    _available_data_types = [t for t in DataType]
    
    for a_data_type in _available_data_types:
        if data_format_name == a_data_type.name:
            _is_data_format_unrecognized = False
            data_type = a_data_type
            
            for sub_type in a_data_type.value:
                
                if data_type_name is not None and isinstance(sub_type, Enum):
                    # check if sub data type exist (it shouldnot if variable is UNKNOWN)
                    if data_type_name == sub_type.name:
                        
                        _is_data_type_unrecognized = False
                        data_type = sub_type
                
                    
                else:
                    _is_data_type_unrecognized = False
    # check for data formt file consistancy error
    if any((_is_data_format_unrecognized, _is_data_type_unrecognized)):
        if _is_data_format_unrecognized:
            raise ValueError(f'error: {data_format_name} not recognized as a valid data format')
        else:
            raise ValueError(f'error {data_type_name} not recognized as a valid data type')
            
    return data_type

def check_data_type_consistancy(data_type: Union[Enum, List[Enum]], d_type: type):
    """ checks if `sub_data_type` folds within """
    ## variable initialization
    is_consistant = False
    
    if not isinstance(data_type, list):
        data_type = [data_type]
    for sub_data_type in data_type:
        for data_type in sub_data_type:
            is_type_in_data_type = any(d_type == t for t in data_type.value)
            if is_type_in_data_type:
                is_consistant = True
                continue
            
    return is_consistant

def get_data_type_propreties(d_type: Enum) -> Enum:
    """Returns data type properties (from `Enum` class DataTypeProperties, 
    regarding the name of the `Enum` class d_type)

    Args:
        d_type (Enum): [description]
    """
    print('data_type', d_type, d_type.name)
    property = None
    for d_type_property in DataTypeProperties:
        print(d_type_property.name, d_type.name)
        if hasattr(d_type_property, d_type.name):
            property = getattr(d_type_property, d_type.name)
            break
    return property
            

def check_missing_data(column: pd.Series)->bool:
    is_missing_data = column.isna().any()
    return is_missing_data


def save_format_file_ref(format_file_ref: Dict[str, Dict[str, Any]], path: str):
    # save `format_file_ref` into a JSON file
    with open(path, "w") as format_file:
        json.dump(format_file_ref, format_file)
    print(f"Format Reference File successfully saved at {path}")

 
def load_format_file_ref(path: str) -> Dict[str, Dict[str, Any]]:
    # retrieve data format file
    with open(path, "r") as format_file:
        format_file_ref = json.load(format_file)
    return format_file_ref

def unique(iterable: Iterator, number: bool = False) -> int:
    """returns number of unique values"""
    set_of_values = set(iterable)
    if number:
        return len(set_of_values)
    else:
        return set_of_values
    
def is_datetime(date: Union[str, int]) -> bool:
    """checks if date is a date"""
    is_date_parsable = True
    
    try:
        if not isinstance(date, str):
            date = str(date)
        dateutil.parser.parse(date)
    except (ParserError, ValueError, TypeError) as err:
        is_date_parsable = False
        
    return is_date_parsable

def get_view_names(view_mapper: Dict[str, Any]) -> List[str]:
    """Gets the keys of a dictionary and returns them into a list.
    Removes from the list names that are not a view"""
    names = list(view_mapper.keys())
    if GLOBAL_THRESHOLDS in names:
        # removes GLOBAL_TRESHOLDS key from list of keys
        # (because it is not a view but general parameters)
        names.remove(GLOBAL_THRESHOLDS)
    return names

def isfloat(value:str) ->bool:
    """checks if string represents a float or int"""
    is_float = True
    try:
        float(value)
    except ValueError as e:
        is_float = False
    return is_float

def is_nan(value: Union[int, float, bool, str]) -> bool:
    """Extends np.isnan method from numpy, check if 
    passed value is a nan or not

    Args:
        value (Union[int, float, bool, str]): [description]

    Returns:
        bool: [description]
    """
    is_value_a_nan = False
    if isinstance(value, str):
        if value.isdecimal():
            value = float(value)
        else:
            if value.lower() in STR_MISSING_VALUE:
                is_value_a_nan = True 
    elif value is None:
        is_value_a_nan = True     
    else:
        is_value_a_nan = np.isnan(value)
    return is_value_a_nan


def infer_type(col: pd.Series, data_format :Enum= None, is_date: bool = False):
    """Completes tabular data type parsing (pandas doesnot parse completly strings
    and datetime colums when loading it)

    Args:
        col (pd.Series): [description]
        data_format (Enum, optional): [description]. Defaults to None.
        is_date (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    col_type = None
        
    if is_date:
        
        try:
            col = pd.to_datetime(col)
            col_type = col.dtype.type
        except (TypeError, ParserError):
            pass
        
    if col_type is None:
        if is_string_dtype(col):
            col_type = str

        elif hasattr(col, 'dtype'):
            col_type = col.dtype
    return col_type
