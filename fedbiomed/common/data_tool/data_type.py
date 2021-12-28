from enum import Enum
import datetime

import numpy as np
import pandas as pd

import aenum


# the use of Enum classes will prevent incorrect combination of values
class QuantitativeDataType(Enum):
    CONTINUOUS = [float, np.float64]
    DISCRETE = [int, np.int64]

class CategoricalDataType(Enum):
    BOOLEAN = [bool]
    NUMERICAL = [float, int, np.float64, np.int64]
    CHARACTER = [str]
    
class KeyDataType(Enum):
    NUMERICAL = [int, np.int64]
    CHARACTER = [str]
    DATETIME = [pd.Timestamp,
                pd.Timedelta,
                pd.Period,
                datetime.datetime,
                np.datetime64]


class CustomDataType(Enum):
    """for demo purpose: here a custom datatype"""
    DISCRETE = [int, np.int64]
    CHARACTER = [str]
    
    
class DataType(Enum):
    """
    Definition of main datatype that will be used for
    data sanity checks
    """
    # what about 
    KEY = KeyDataType
    QUANTITATIVE = QuantitativeDataType
    CATEGORICAL = CategoricalDataType
    DATETIME = [pd.Timestamp,
                pd.Timedelta,
                pd.Period,
                datetime.datetime,
                np.datetime64]
    CUSTOM = CustomDataType  # custom data type (should be defined by user)
    UNKNOWN = 'UNKNOWN'
    
    @staticmethod
    def get_names():
        return tuple(n for n, _ in DataType.__members__.items())
    
class DataTypeProperties(Enum):
    """Data Type possible modification (whithin CLI editing)"""
    CATEGORICAL = (False, False, True, False,  False, True)
    QUANTITATIVE = (True, True, False, False, False, True)
    DATETIME = (True, True, False, True, False, False)
    UNKNOWN = (False, False, False, False, False, True)
    CUSTOM = (True, True, True, False, False, True)
    KEY = (True, True, False, True, True, False)

    def __init__(self,
                 lower_bound: bool,
                 upper_bound: bool,
                 set_of_values: bool,
                 date_format: bool,
                 date_ambiguity:bool,
                 allow_missing_values: bool):
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._set_of_values = set_of_values
        self._date_ambiguity = date_ambiguity
        self._date_format = date_format
        self._allow_missing_values = allow_missing_values
    
    @property
    def lower_bound(self):
        return self._lower_bound
    
    @property
    def upper_bound(self):
        return self._upper_bound
    
    @property
    def set_of_values(self):
        return self._set_of_values
    
    @property
    def date_format(self):
        return self._date_format
    
    @property
    def date_ambiguity(self):
        return self._date_ambiguity

    @property
    def allow_missing_values(self):
        return self._allow_missing_values


def extend_data_type_properties(name, value):
    aenum.extend_enum(DataTypeProperties, name, value)

def extend_data_type(name, value):
    aenum.extend_enum(DataType, name, value)