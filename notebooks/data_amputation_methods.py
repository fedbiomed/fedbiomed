
from enum import Enum
from functools import partial
from typing import Union, Callable, List
from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np
from data_type import QuantitativeDataType, CategoricalDataType

# imputation methods

def impute_missing_values_mean(data):
    try:
        if type(data) == pd.core.frame.DataFrame:
            for col in data.columns:
                if (data[col].isnull().sum()>0):
                    if any(data[col].dtype in x2 for x2 in  [x.value for x in QuantitativeDataType]):
                        data[col].fillna(value=data[col].mean(),inplace=True)
        else:
            data = data.fillna(data.mean())
        return data
    except Exception as err:
        print(err)
        print('Error encountered in loading data file')
    
 #Categorical Data
def impute_missing_values_mode(data: Union[pd.Series, pd.DataFrame]):
    try:
        if type(data) == pd.core.frame.DataFrame:
            for col in data.columns:
                if (data[col].isnull().sum()>0):
                    categorical_data_type = [x.value for x in CategoricalDataType]
                    data_type_condition = any(data[col].dtype in x2 for x2 in categorical_data_type)
                    if data_type_condition:
                        print(col)
                        data[col].fillna(value=data[col].mode()[0],inplace=True)
                        
        else:
            data = data.fillna(data.value_counts().index[0])
            
        return data
    except Exception as err:
        print(err)
        print('Error encountered in imputing missing values - mode')
        
    # Impute missing values with KNN            

def impute_missing_values_knn(data,k=2):
    try:
        if type(data) == pd.core.frame.DataFrame:
            missing_cols = data.columns[data.isnull().any()]
            if len(missing_cols)>0:
                    imputer =KNNImputer(n_neighbors=k)
                    data = pd.DataFrame(imputer.fit_transform(data),columns=data.columns) 
        else:
            imputer =KNNImputer(n_neighbors=k)
            data =pd.DataFrame( (imputer.fit_transform(np.array(data).reshape(1,-1))).reshape(-1,1),columns=[data.name])                    

        return data  
    except Exception as err:
        print(err)
        print('Error encountered in imputing missing values - knn')

#Impute missing values with Interpolate
def impute_missing_values_interpolate(data):
    try:
        data_filled = data.interpolate()
            
        return data_filled
    except Exception as err:
        print(err)
        print('Error encountered in imputing missing values - interpolate')
        
class ImputationMethods(Enum):
    MEAN_IMPUTATION = (partial(impute_missing_values_mean),
                       QuantitativeDataType, None)
    MODE_IMPUTATION = (partial(impute_missing_values_mode),
                       CategoricalDataType, None)
    KNN_IMPUTATION = (partial(impute_missing_values_knn),
                      CategoricalDataType, ['k'])
    INTERPOLATION_IMPUTATION = (partial(impute_missing_values_interpolate),
                                QuantitativeDataType, None)
    
    def __init__(self, method: Callable,
                 data_type: Enum,
                 parameters_to_ask_user: List[str]):
        self._method = method
        self._data_type = data_type
        self._parameters_to_ask_user = parameters_to_ask_user
    
    def __call__(self, *args):
        """method avoiding to specify `value` when using an enum class"""
        self.value(*args)

    
    def method(self, *args):
        val = self._method(*args)
        return val
    
    @property
    def data_type(self):
        return self._data_type
    
    @property
    def parameters_to_ask_user(self):
        return self._parameters_to_ask_user