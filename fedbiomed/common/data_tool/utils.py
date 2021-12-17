from enum import Enum
import pandas as pd
import csv
import json

from typing import List, Tuple, Union, Dict, Any, Iterator, Optional, Callable
import os
import dateutil
from dateutil.parser._parser import ParserError

from fedbiomed.common.data_tool.data_type import DataType


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
                  #avail_data_types: enum.EnumMeta,
                  d_format: Enum,
                  d_type: type) ->  Tuple[Enum, List[Union[type, str]]]:
    # varibales initialisation
    present_d_types = []
    sub_d_type_format = d_format
    
    
    for avail_data_type in DataType:
        if d_format is avail_data_type:
            sub_dtypes = avail_data_type.value
            #if not isinstance(sub_dtypes, str) and hasattr(sub_dtypes, '__getitem__') and isinstance(sub_dtypes[0], Enum):
            if not isinstance(sub_dtypes, str):
                # check if dtype has subtypes
                #(eg if datatype is QUANTITATIVE, subtype will be CONTINOUS or DISCRETE)
                if isinstance(next(iter(sub_dtypes)), Enum):
                    
                    for sub_dtype in sub_dtypes:
                        if any(d_type == t for t in tuple(sub_dtype.value)):
                            present_d_types.append(d_type)
                            sub_d_type_format = sub_dtype
                            print(sub_dtype, d_type)
                else:
                    # case where datatype doesnot have subtypes, eg DATETIME
                    if any(d_type == t for t in sub_dtypes):
                        present_d_types.append(d_type)
                    sub_d_type_format = d_format
            else:
                # case where d_format is a string of character
                sub_d_type_format = d_format
    print(sub_d_type_format, '|', present_d_types)
    return  sub_d_type_format, present_d_types


def find_data_type(data_format_name: str, data_type_name: str=None) -> Enum:
    """Retrieves from a given data_format and data_type,
    the corresponding Enum class describing data"""
    
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
                
                if data_type_name is not None and  isinstance(sub_type, Enum):
                    # check if sub data type exist (it shouldnot if variable is UNKNOWN)
                    if data_type_name == sub_type.name:
                        
                        _is_data_type_unrecognized = False
                        data_type = sub_type
                
                    
                else:
                    _is_data_type_unrecognized = False
    # check for data formt file consistancy error
    if any((_is_data_format_unrecognized, _is_data_type_unrecognized)):
        if _is_data_format_unrecognized:
            raise ValueError(f'error: {data_format_name} not recognized as a valid data type')
        else:
            raise ValueError(f'error {data_type_name} not recognized as a valid data type')
            
    return data_type

def check_data_type_consistancy(sub_data_type: Enum, d_type: type):
    """ checks if `sub_data_type` folds within """
    is_consistant = False
    for data_type in sub_data_type:
        is_type_in_data_type = any(d_type == t for t in data_type.value)
        if is_type_in_data_type:
            is_consistant = True
            continue
            
    return is_consistant

def check_missing_data(column: pd.Series)->bool:
    is_missing_data = column.isna().any()
    return is_missing_data


def save_format_file_ref(format_file_ref: Dict[str, Dict[str, Any]], path: str):
    # save `format_file_ref` into a JSON file
    with open(path, "w") as format_file:
        json.dump(format_file_ref, format_file)
    print(f"Model successfully saved at {path}")
    
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
    
def is_datetime(date: str) -> bool:
    """checks if date is a date"""
    is_date_parsable = True
    try:
        dateutil.parser.parse(date)
    except (ParserError, ValueError) as err:
        is_date_parsable = False
        
    return is_date_parsable
