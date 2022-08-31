import uuid
from typing import Any, Dict, List, Tuple, TypeVar, Union
import re
from abc import ABC, abstractmethod

from fedbiomed.common.exceptions import FedbiomedLoadingBlockValueError, FedbiomedDataLoadingPlanValueError, \
    FedbiomedLoadingBlockError, FedbiomedDataLoadingPlanError
from fedbiomed.common.logger import logger
from fedbiomed.common.constants import ErrorNumbers, DataLoadingBlockTypes, DatasetTypes
from fedbiomed.common.validator import SchemeValidator, ValidatorError, \
    ValidateError, RuleError, validator_decorator


TDataLoadingPlan = TypeVar("TDataLoadingPlan", bound="DataLoadingPlan")
TDataLoadingBlock = TypeVar("TDataLoadingBlock", bound="DataLoadingBlock")


class SerializedDataLoadingBlockValidation:
    def __init__(self):
        self._validation_scheme = SerializedDataLoadingBlockValidation.default_scheme()

    def validate_serialized_dlb(self,
                                dlb_metadata: Dict,
                                only_required: bool = True):
        try:
            sc = SchemeValidator(self._validation_scheme)
        except RuleError as e:
            msg = ErrorNumbers.FB614.value + f": {e}"
            logger.critical(msg)
            raise FedbiomedLoadingBlockValueError(msg)

        try:
            dlb_metadata = sc.populate_with_defaults(dlb_metadata,
                                                     only_required=only_required)
        except ValidatorError as e:
            msg = ErrorNumbers.FB614.value + f": {e}"
            logger.critical(msg)
            raise FedbiomedLoadingBlockValueError(msg)

        try:
            sc.validate(dlb_metadata)
        except ValidateError as e:
            msg = ErrorNumbers.FB614.value + f": {e}"
            logger.critical(msg)
            raise FedbiomedLoadingBlockValueError(msg)

    @staticmethod
    @validator_decorator
    def _identifier_validation_hook(classname: str) -> Union[bool, Tuple[bool, str]]:
        for name in classname.split('.'):
            if not re.match('^[_a-zA-Z]\w*$', name):
                return False, f'{name} within {classname} is not a valid class name ' \
                              f'for deserialization of Data Loading Block.'
        return True

    @staticmethod
    @validator_decorator
    def _serial_id_validation_hook(serial_id: str) -> Union[bool, Tuple[bool, str]]:
        if serial_id[:15] != 'serialized_dlb_':
            return False, f'{serial_id} is not of the form serialized_dlb_<uuid> ' \
                          f'for deserialization of Data Loading Block.'
        try:
            uuid_obj = uuid.UUID(serial_id[15:])
        except ValueError:
            return False, f'{serial_id} is not of the form serialized_dlb_<uuid> ' \
                          f'for deserialization of Data Loading Block.'
        return True

    @classmethod
    def default_scheme(cls) -> Dict:
        return {
            'loading_block_class': {
                'rules': [str, cls._identifier_validation_hook],
                'required': True,
            },
            'loading_block_module': {
                'rules': [str, cls._identifier_validation_hook],
                'required': True,
            },
            'loading_block_serialization_id': {
                'rules': [str, cls._serial_id_validation_hook],
                'required': True,
            },
        }


class DataLoadingBlock(SerializedDataLoadingBlockValidation, ABC):
    """The building blocks of a DataLoadingPlan.

    A DataLoadingBlock describes an intermediary layer between the researcher
    and the node's filesystem. It allows the node to specify a customization
    in the way data is "perceived" by the data loaders during training.

    A DataLoadingBlock is identified by its type_id attribute. Thus this
    attribute should be unique among all DataLoadingBlockTypes in the same
    DataLoadingPlan. Moreover, we may test equality between a
    DataLoadingBlock and a string by checking its type_id, as a means of
    easily testing whether a DataLoadingBlock is contained in a collection.

    Correct usage of this class requires creating ad-hoc subclasses.
    The DataLoadingBlock class is not intended to be instantiated directly.

    Subclasses of DataLoadingBlock must respect the following conditions:
    1. implement a constructor taking exactly one argument, a type_id string
    2. the implemented constructor must call super().__init__(type_id)
    3. extend the serialize(self) and the deserialize(self, load_from: dict) functions
    4. both serialize and deserialize must call super's serialize and deserialize respectively
    5. the deserialize function must always return self
    6. the serialize function must update the dict returned by super's serialize
    7. implement an apply function that takes arbitrary arguments and applies
       the logic of the loading_block

    Attributes:
        __serialization_id: (str) identifies *one serialized instance* of the DataLoadingBlock
    """
    def __init__(self):
        super().__init__()
        self.__serialization_id = 'serialized_dlb_' + str(uuid.uuid4())

    def get_serialization_id(self):
        """Expose serialization id as read-only"""
        return self.__serialization_id

    def serialize(self) -> dict:
        """Serializes the class in a format similar to json.

        Returns:
             a dictionary of key-value pairs sufficient for reconstructing
             the DataLoadingBlock.
        """
        return dict(
            loading_block_class=self.__class__.__qualname__,
            loading_block_module=self.__module__,
            loading_block_serialization_id=self.__serialization_id
        )

    def deserialize(self, load_from: dict) -> TDataLoadingBlock:
        """Reconstruct the DataLoadingBlock from a serialized version.

        Args:
            load_from: a dictionary as obtained by the serialize function.
        Returns:
            the self instance
        """
        self.validate_serialized_dlb(load_from)
        self.__serialization_id = load_from['loading_block_serialization_id']
        return self

    @abstractmethod
    def apply(self, *args, **kwargs):
        """Abstract method representing an application of the DataLoadingBlock
        """
        pass

    @staticmethod
    def instantiate_class(loading_block: dict) -> TDataLoadingBlock:
        """Instantiate one DataLoadingBlock object of the type defined in the arguments.

        Uses the `loading_block_module` and `loading_block_class` fields of the loading_block argument to
        identify the type of DataLoadingBlock to be instantiated, then calls its default constructor.
        Note that this function **does not call deserialize**.

        Args:
            loading_block: a dictionary of DataLoadingBlock metadata in the format returned by the serialize function.
        Returns:
            A default-constructed instance of a DataLoadingBlock of the type defined in the metadata.
        Raises:
           FedbiomedLoadingBlockError: if the instantiation process raised any exception.
        """
        try:
            exec(f"import {loading_block['loading_block_module']}")
            dlb = eval(f"{loading_block['loading_block_module']}.{loading_block['loading_block_class']}()")
        except Exception as e:
            raise FedbiomedLoadingBlockError(f"Could not instantiate DataLoadingBlock from the following metadata: "
                                             f"{loading_block} because of {type(e).__name__}: {e}")
        return dlb


class MapperBlock(DataLoadingBlock):
    """A DataLoadingBlock for mapping values.

    This Dataloading_block can be used whenever an "indirect mapping" is needed.
    For example, it can be used to implement a correspondence between a set
    of "logical" abstract names and a set of folder names on the filesystem.

    The apply function of this DataLoadingBlock takes a "key" as input (a str)
    and returns the mapped value corresponding to map[key].
    Note that while the constructor of this class sets a value for type_id,
    developers are recommended to set a more meaningful value that better
    speaks to their application.

    Multiple instances of this loading_block may be used in the same DataLoadingPlan,
    provided that they are given different type_id via the constructor.
    """
    def __init__(self):
        super(MapperBlock, self).__init__()
        self.map = {}
        self._validation_scheme.update(MapperBlock._extra_validation_scheme())

    def serialize(self) -> dict:
        """Serializes the class in a format similar to json.

        Returns:
             a dictionary of key-value pairs sufficient for reconstructing
             the DataLoadingBlock.
        """
        ret = super(MapperBlock, self).serialize()
        ret.update({'map': self.map})
        return ret

    def deserialize(self, load_from: dict) -> DataLoadingBlock:
        """Reconstruct the DataLoadingBlock from a serialized version.

        Args:
            load_from: a dictionary as obtained by the serialize function.
        Returns:
            the self instance
        """
        super(MapperBlock, self).deserialize(load_from)
        self.map = load_from['map']
        return self

    def apply(self, key):
        if not isinstance(self.map, dict) or key not in self.map:
            raise FedbiomedLoadingBlockError(f"Mapper block error: no key '{key}' in mapping dictionary")
        return self.map[key]

    @classmethod
    def _extra_validation_scheme(cls):
        return {
            'map': {
                'rules': [dict],
                'required': True
            }
        }


class SerializedDataLoadingPlanValidation:
    def __init__(self):
        self._validation_scheme = SerializedDataLoadingPlanValidation.default_scheme()

    def validate_serialized_dlp(self,
                                dlb_metadata: Dict,
                                only_required: bool = True):
        try:
            sc = SchemeValidator(self._validation_scheme)
        except RuleError as e:
            msg = ErrorNumbers.FB614.value + f": {e}"
            logger.critical(msg)
            raise FedbiomedDataLoadingPlanValueError(msg)

        try:
            dlb_metadata = sc.populate_with_defaults(dlb_metadata,
                                                     only_required=only_required)
        except ValidatorError as e:
            msg = ErrorNumbers.FB614.value + f": {e}"
            logger.critical(msg)
            raise FedbiomedDataLoadingPlanValueError(msg)

        try:
            sc.validate(dlb_metadata)
        except ValidateError as e:
            msg = ErrorNumbers.FB614.value + f": {e}"
            logger.critical(msg)
            raise FedbiomedDataLoadingPlanValueError(msg)

    @staticmethod
    @validator_decorator
    def _target_dataset_type_validator(dataset_type: str) -> Union[bool, Tuple[bool, str]]:
        return dataset_type in [t.value for t in DatasetTypes]

    @staticmethod
    @validator_decorator
    def _loading_blocks_types_validator(loading_blocks: dict) -> Union[bool, Tuple[bool, str]]:
        if not isinstance(loading_blocks, dict):
            return False, f"Field loading_blocks must be of type dict, instead found {type(loading_blocks).__name__}"
        for key, value in loading_blocks.items():
            if key not in [k.value for child in DataLoadingBlockTypes.__subclasses__() for k in child]:
                return False, f"Data loading block key {key} is not a valid key."
            if not isinstance(value, str):
                return False, f"Data loading block id {value} is not valid."
        return True

    @staticmethod
    @validator_decorator
    def _key_paths_validator(key_paths: dict) -> Union[bool, Tuple[bool, str]]:
        if not isinstance(key_paths, dict):
            return False, f"Field key_paths must be of type dict, instead found {type(key_paths).__name__}"
        for key, value in key_paths.items():
            if key not in [k.value for child in DataLoadingBlockTypes.__subclasses__() for k in child]:
                return False, f"Data loading block key {key} is not a valid key."
            if not isinstance(value, tuple):
                return False, f"Values for the key_paths dictionary should be tuples, " \
                              f"instead found {type(value).__name__}."
            if len(value) != 2:
                return False, f"Values for the key_paths dictionary should have length 2, instead found {len(value)}."
            if not isinstance(value[0], str) or not isinstance(value[1], str):
                return False, f"Key paths should be (str, str) tuples, instead found ({type(value[0]).__name__}, " \
                              f"{type(value[1]).__name__})."
        return True

    @classmethod
    def default_scheme(cls) -> Dict:
        return {
            'dlp_id': {
                'rules': [str],
                'required': True,
            },
            'dlp_name': {
                'rules': [str],
                'required': True,
            },
            'target_dataset_type': {
                'rules': [str, cls._target_dataset_type_validator],
                'required': True,
            },
            'loading_blocks': {
                'rules': [dict, cls._loading_blocks_types_validator],
                'required': True
            },
            'key_paths': {
                'rules': [dict, cls._key_paths_validator],
                'required': True
            }
        }


class DataLoadingPlan(Dict[DataLoadingBlockTypes, DataLoadingBlock], SerializedDataLoadingPlanValidation):
    """Customizations to the way the data is loaded and presented for training.

    A DataLoadingPlan is a dictionary of {name: DataLoadingBlock} pairs. Each
    DataLoadingBlock represents a customization to the way data is loaded and
    presented to the researcher. These customizations are defined by the node,
    but they operate on a Dataset class, which is defined by the library and
    instantiated by the researcher.

    To exploit this functionality, a Dataset must be modified to accept the
    customizations provided by the DataLoadingPlan. To simplify this process,
    we provide the DataLoadingPlanMixin class below.

    The DataLoadingPlan class should be instantiated directly, no subclassing
    is needed. The DataLoadingPlan *is* a dict, and exposes the same interface
    as a dict.

    Attributes:
        dlp_id: str representing a unique plan id (auto-generated)
        desc: str representing an optional user-friendly short description
        target_dataset_type: a DatasetTypes enum representing the type of dataset targeted by this DataLoadingPlan
    """
    def __init__(self, *args, **kwargs):
        super(DataLoadingPlan, self).__init__(*args, **kwargs)
        super(dict, self).__init__()
        self.dlp_id = 'dlp_' + str(uuid.uuid4())
        self.desc = ""
        self.target_dataset_type = DatasetTypes.NONE

    def __setitem__(self, key, value):
        """Type-check the arguments then call dict.__setitem__."""
        if not isinstance(key, DataLoadingBlockTypes):
            raise FedbiomedDataLoadingPlanValueError(f"Key {key} is not of enum type DataLoadingBlockTypes in"
                                                     f" DataLoadingPlan {self}")
        if not isinstance(value, DataLoadingBlock):
            raise FedbiomedDataLoadingPlanValueError(f"Value {value} is not of type DataLoadingBlock in"
                                                     f" DataLoadingPlan {self}")
        super().__setitem__(key, value)

    def serialize(self) -> Tuple[dict, List]:
        """Serializes the class in a format similar to json.

        Returns:
             a tuple sufficient for reconstructing the DataLoading plan. It includes:
             - a dictionary of key-value pairs with the DataLoadingPlan parameters.
             - a list of dict containing the data for reconstruction all the DataLoadingBlock
               of the DataLoadingPlan 
        """
        return dict(
            dlp_id=self.dlp_id,
            dlp_name=self.desc,
            target_dataset_type=self.target_dataset_type.value,
            loading_blocks={key.value: dlb.get_serialization_id() for key, dlb in self.items()},
            key_paths={key.value: (f"{key.__module__}", f"{key.__class__.__qualname__}") for key in self.keys()}
        ), [dlb.serialize() for dlb in self.values()]

    def deserialize(self, serialized_dlp: dict, serialized_loading_blocks: List[dict]) -> TDataLoadingPlan:
        """Reconstruct the DataLoadingPlan from a serialized version.

        The format of the input argument is expected to be an 'aggregated serialized' version, as defined by the output
        of the 'DataLoadingPlan.aggregate_serialized_metadata` function.

        :warning: Calling this function will *clear* the contained
            DataLoadingBlockTypes. This function may not be used to "update"
            nor to "append to" a DataLoadingPlan.

        Args:
            serialized_dlp: a dictionary of data loading plan metadata, as obtained from the first output of the
                serialize function
            serialized_loading_blocks: a list of dictionaries of loading_block metadata, as obtained from the second
                output of the serialize function
        Returns:
            the self instance
        """
        self.validate_serialized_dlp(serialized_dlp)

        self.clear()
        self.dlp_id = serialized_dlp['dlp_id']
        self.desc = serialized_dlp['dlp_name']
        self.target_dataset_type = DatasetTypes(serialized_dlp['target_dataset_type'])
        for loading_block_key_str, loading_block_serialization_id in serialized_dlp['loading_blocks'].items():
            key_module, key_classname = serialized_dlp['key_paths'][loading_block_key_str]
            try:
                exec(f"import {key_module}")
                loading_block_key = eval(f"{key_module}.{key_classname}('{loading_block_key_str}')")
            except Exception as e:
                raise FedbiomedDataLoadingPlanError(f"Error deserializing loading block key "
                                                    f"{loading_block_key_str} with path {key_module}.{key_classname} "
                                                    f"because of {type(e).__name__}: {e}")
            loading_block = next(filter(lambda x: x['loading_block_serialization_id'] == loading_block_serialization_id,
                                        serialized_loading_blocks))
            dlb = DataLoadingBlock.instantiate_class(loading_block)
            self[loading_block_key] = dlb.deserialize(loading_block)
        return self

    def __str__(self):
        """User-friendly string representation"""
        return f"Data Loading Plan {self.desc} id: {self.dlp_id} "\
               f"containing: {'; '.join([k.value for k in self.keys()])}"

    @staticmethod
    def infer_dataset_type(dataset: Any) -> DatasetTypes:
        """Infer the type of a given dataset.

        This function provides the mapping between a dataset's class and the DatasetTypes enum. If the dataset exposes
        the correct interface (i.e. the get_dataset_type method) then it directly calls that, otherwise it tries to
        apply some heuristics to guess the type of dataset.

        Args:
            dataset: the dataset whose type we want to infer.
        Returns:
            a DatasetTypes enum value which identifies the type of the dataset.
        Raises:
            FedbiomedDataLoadingPlanValueError: if the dataset does not have a `get_dataset_type` method and moreover
                the type could not be guessed.
        """
        if hasattr(dataset, 'get_dataset_type'):
            return dataset.get_dataset_type()
        elif dataset.__class__.__name__ == 'ImageFolder':
            # ImageFolder could be both an images type or mednist. Try to identify mednist with some heuristic.
            if hasattr(dataset, 'classes') and \
                    all([x in dataset.classes for x in ['AbdomenCT', 'BreastMRI', 'CXR', 'ChestCT', 'Hand', 'HeadCT']]):
                return DatasetTypes.MEDNIST
            else:
                return DatasetTypes.IMAGES
        elif dataset.__class__.__name__ == 'MNIST':
            return DatasetTypes.DEFAULT
        raise FedbiomedDataLoadingPlanValueError(f"Trying to infer dataset type of {dataset} is not supported "
                                                 f"for datasets of type {dataset.__class__.__qualname__}")


class DataLoadingPlanMixin:
    """Utility class to enable DLP functionality in a dataset.

    Any Dataset class that inherits from DataLoadingPlanMixin will have the
    basic tools necessary to support a DataLoadingPlan. Typically, the logic
    of each specific DataLoadingBlock in the DataLoadingPlan will be implemented
    in the form of hooks that are called within the Dataset's implementation
    using the helper function apply_dlb defined below.
    """
    def __init__(self):
        self._dlp = None

    def set_dlp(self, dlp: DataLoadingPlan):
        """Sets the dlp if the target dataset type is appropriate"""
        if not isinstance(dlp, DataLoadingPlan):
            raise FedbiomedDataLoadingPlanValueError(f"Trying to set a DataLoadingPlan but the argument is of type "
                                                     f"{type(dlp).__name__}")

        dataset_type = DataLoadingPlan.infer_dataset_type(self)  # `self` here will refer to the Dataset instance
        if dlp.target_dataset_type != DatasetTypes.NONE and dataset_type != dlp.target_dataset_type:
            raise FedbiomedDataLoadingPlanValueError(f"Trying to set {dlp} on dataset of type {dataset_type.value} but "
                                                     f"the target type is {dlp.target_dataset_type}")
        elif dlp.target_dataset_type == DatasetTypes.NONE:
            dlp.target_dataset_type = dataset_type
        self._dlp = dlp

    def clear_dlp(self):
        self._dlp = None

    def apply_dlb(self, default_ret_value: Any, dlb_key: DataLoadingBlockTypes, *args, **kwargs):
        """Apply one DataLoadingBlock identified by its key.

        Note that we want to easily support the case where the DataLoadingPlan
        is not activated, or the requested loading block is not contained in the
        DataLoadingPlan. This is achieved by providing a default return value
        to be returned when the above conditions are met. Hence, most of the
        calls to apply_dlb will look like this:
        ```
        value = self.apply_dlb(value, 'my-loading-block', my_apply_args)
        ```
        This will ensure that value is not changed if the DataLoadingPlan is
        not active.

        Args:
            default_ret_value: the value to be returned in case that the dlp
                functionality is not required
            dlb_key: the key of the DataLoadingBlock to be applied
            args: forwarded to the DataLoadingBlock's apply function
            kwargs: forwarded to the DataLoadingBlock's apply function
        Returns:
             the output of the DataLoadingBlock's apply function, or
             the default_ret_value when dlp is None or it does not contain
             the requested loading block
        """
        if not isinstance(dlb_key, DataLoadingBlockTypes):
            raise FedbiomedDataLoadingPlanValueError(f"Key {dlb_key} is not of enum type DataLoadingBlockTypes"
                                                     f" in DataLoadingPlanMixin.apply_dlb")
        if self._dlp is not None and dlb_key in self._dlp:
            return self._dlp[dlb_key].apply(*args, **kwargs)
        else:
            return default_ret_value




