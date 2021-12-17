    """Definition of the checks performed as data sanity checks

    """
from enum import Enum
from typing import Union, List
import fedbiomed.common.data_tool.pre_processing_warnings_exceptions as check_exception

    
class PreProcessingChecks(Enum):
    INCORRECT_FORMAT_FILE = ("Format File %s is incorrect: cannot parse variable %s", WarningType.CRITICAL_WARNING,
                            )
    KEY_UNICITY_VIOLATED = ("Key Variable %s violated unicity of data", check_exception.WarningType.CRITICAL_WARNING, 
                           )
    MISSING_DATA_NOT_ALLOWED = ("Variable %s must not have missing data, but some were found",
                                check_exception.MissingDataException, 
                               )
    MISSING_DATA_ALLOWED = ("Missing data found in variable %s", check_exception.WarningType.REGULAR_WARNING 
                           )
    INCORRECT_STRUCTURE_DATA_TYPE = ("Data Type %s has an incorrect structure: %s",
                                     check_exception.WarningType.CRITICAL_WARNING)
    DATA_TYPE_MISMATCH = ("Data Type  %s mismatch: %s is not a subtype of %s",
                           check_exception.WarningType.REGULAR_WARNING)
    
    INCORRECT_DATA_TYPE = ('Variable named %s should be a %s variable, but it contains %s type',
                          check_exception.WarningType.REGULAR_WARNING)
    INCORRECT_DATETIME_DATA = ("Variable %s has been defined as a DATETIME variable, but samples are not parsable as date",
                               check_exception.WarningType.CRITICAL_WARNING)
    
    OUTLIER_DETECTION_LOWER_BOUND = ("Detected outliers for Variable %s: samples violate lower bound %s",
                                   check_exception.WarningType.CRITICAL_WARNING)
    
    OUTLIER_DETECTION_UPPER_BOUND = ("Detected outliers for Varaiable %s: samples violate upper bound %s",
                                   check_exception.WarningType.CRITICAL_WARNING)
    
    INCORRECT_VALUES_CATEGORICAL_DATA = ("Found at least one sample with incorrect label in Categorical Vraiable %s. Expected data are %s, but found %s",
                                        check_exception.WarningType.CRITICAL_WARNING)
    
    N_MISSING_DATA_ABOVE_THRESHOLD = ("Found too many missing samples in variable %s, threshold is set at %s",
                                     check_exception.WarningType.CRITICAL_WARNING)
    
    N_SAMPLES_BELOW_THRESHOLD = ("Number of samples contained in dataset %s is below threshold (expected at least %s samples, found %s samples)",
                                check_exception.MinimumSamplesViolatedException)
    MISSING_FEATURE = ("Feature %s has not been found in dataset, but is needed for experiment",
                       check_exception.MissingFeatureException)
    
    #MISSING_VIEW = ("View %s not found in dataset, but needed for experiment")
    
    def __init__(self, message: str,  warning_type: Union[check_exception.WarningType, Exception]):
        self._message = message
        #self._additional_message = additional_message
        self._warning_type = warning_type
        #self._is_exception = is_exception
        
    @property
    def error_message(self):
        return self._message

    @property
    def warning_type(self):
        return self._warning_type
    
    @property
    def additional_message(self):
        return self._additional_message
    
    def __call__(self, *kwargs) -> Union[str, Exception]:
        
        msg = self.error_message % kwargs
        if isinstance(self.warning_type, check_exception.WarningType):
            
            return msg
        elif issubclass(self.warning_type, Exception):
            
            return self.warning_type(message=msg)
        
