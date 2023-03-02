# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import uuid
from typing import Any, Dict, List, Tuple, TypeVar, Type, Union, Optional
from abc import ABC, abstractmethod
from importlib import import_module

from fedbiomed.common.exceptions import FedbiomedError, FedbiomedLoadingBlockValueError, \
    FedbiomedDataLoadingPlanValueError, FedbiomedLoadingBlockError, FedbiomedDataLoadingPlanError
from fedbiomed.common.logger import logger
from fedbiomed.common.constants import ErrorNumbers, DataLoadingBlockTypes, DatasetTypes
from fedbiomed.common.validator import SchemeValidator, ValidatorError, \
    ValidateError, RuleError, validator_decorator

TDataLoadingPlan = TypeVar("TDataLoadingPlan", bound="DataLoadingPlan")
TDataLoadingBlock = TypeVar("TDataLoadingBlock", bound="DataLoadingBlock")


class SerializationValidation:
    """Provide Validation capabilities for serializing/deserializing a [DataLoadingBlock] or [DataLoadingPlan].

    When a developer inherits from [DataLoadingBlock] to define a custom loading block, they are required to call
    the `_serialization_validator.update_validation_scheme` function with a dictionary argument containing the
    rules to validate all the additional fields that will be used in the serialization of their loading block.

    These rules must follow the syntax explained in the [SchemeValidator][fedbiomed.common.validator.SchemeValidator]
    class.

    For example
    ```python
        class MyLoadingBlock(DataLoadingBlock):
            def __init__(self):
                self.my_custom_data = {}
                self._serialization_validator.update_validation_scheme({
                    'custom_data': {
                        'rules': [dict, ...any other rules],
                        'required': True
                    }
                })
            def serialize(self):
                serialized = super().serialize()
                serialized.update({'custom_data': self.my_custom_data})
                return serialized
    ```

    Attributes:
       _validation_scheme: (dict) an extensible set of rules to validate the DataLoadingBlock metadata.
    """

    def __init__(self):
        self._validation_scheme = {}

    def validate(self,
                 dlb_metadata: Dict,
                 exception_type: Type[FedbiomedError],
                 only_required: bool = True) -> None:
        """Validate a dict of dlb_metadata according to the _validation_scheme.

        Args:
            dlb_metadata (dict) : the [DataLoadingBlock] metadata, as returned by serialize or as loaded from the
                node database.
            exception_type (Type[FedbiomedError]): the type of the exception to be raised when validation fails.
            only_required (bool) : see SchemeValidator.populate_with_defaults
        Raises:
            exception_type: if the validation fails.
        """
        try:
            sc = SchemeValidator(self._validation_scheme)
        except RuleError as e:
            msg = ErrorNumbers.FB614.value + f": {e}"
            logger.critical(msg)
            raise exception_type(msg)

        try:
            dlb_metadata = sc.populate_with_defaults(dlb_metadata,
                                                     only_required=only_required)
        except ValidatorError as e:
            msg = ErrorNumbers.FB614.value + f": {e}"
            logger.critical(msg)
            raise exception_type(msg)

        try:
            sc.validate(dlb_metadata)
        except ValidateError as e:
            msg = ErrorNumbers.FB614.value + f": {e}"
            logger.critical(msg)
            raise exception_type(msg)

    def update_validation_scheme(self, new_scheme: dict) -> None:
        """Updates the validation scheme.

        Args:
            new_scheme: (dict) new dict of rules
        """
        self._validation_scheme.update(new_scheme)

    @staticmethod
    @validator_decorator
    def _identifier_validation_hook(full_name: str) -> Union[bool, Tuple[bool, str]]:
        """Validates that a fully qualified name follows the syntax for python identifiers.

        Valid identifiers are of the form (in regexp) "^[a-zA-Z_]\w*$" and multiple identifiers may be combined with
        dots in a fully qualified name in case of inheritance.

        Args:
            full_name (str): the fully qualified name, composed of either one identifier (a class or module name) or
                multiple identifiers separated by a dot.
        Returns:
            True if the full name is valid, or a tuple (False, str message) otherwise
        """
        for name in full_name.split('.'):
            if not name.isidentifier():
                return False, f'{name} within {full_name} is not a valid identifier ' \
                              f'for deserialization of Data Loading Block.'
        return True

    @staticmethod
    @validator_decorator
    def _serial_id_validation_hook(serial_id: str) -> Union[bool, Tuple[bool, str]]:
        """Validates the syntax of a DataLoadingBlock's serial id.

        The serial id must be of the form:
            serialized_dlb_<UUID>

        Args:
            serial_id (str): the id to validate.
        Returns:
            True if the serial id is valid, or a tuple (False, str message) otherwise
        """
        if serial_id[:15] != 'serialized_dlb_':
            return False, f'{serial_id} is not of the form serialized_dlb_<uuid> ' \
                          f'for deserialization of Data Loading Block.'
        try:
            _ = uuid.UUID(serial_id[15:])
        except ValueError:
            return False, f'{serial_id} is not of the form serialized_dlb_<uuid> ' \
                          f'for deserialization of Data Loading Block.'
        return True

    @staticmethod
    @validator_decorator
    def _target_dataset_type_validator(dataset_type: str) -> Union[bool, Tuple[bool, str]]:
        """Validate that the target type of a [DataLoadingPlan] is a valid enum value."""
        return dataset_type in [t.value for t in DatasetTypes]

    @staticmethod
    @validator_decorator
    def _loading_blocks_types_validator(loading_blocks: dict) -> Union[bool, Tuple[bool, str]]:
        """Validate that loading blocks is a dict of {DataLoadingBlockTypes: str}."""
        if not isinstance(loading_blocks, dict):
            return False, f"Field loading_blocks must be of type dict, instead found {type(loading_blocks).__name__}"
        for key, value in loading_blocks.items():
            # Developers may inherit from DataLoadingBlockTypes to define their own types within the scope of their
            # implementation. Hence, we must search through the subclasses to get all possible values.
            if key not in [k.value for child in DataLoadingBlockTypes.__subclasses__() for k in child]:
                return False, f"Data loading block key {key} is not a valid key."
            if not isinstance(value, str):
                return False, f"Data loading block id {value} is not valid."
        return True

    @staticmethod
    @validator_decorator
    def _key_paths_validator(key_paths: Dict[DataLoadingBlockTypes, Union[tuple, list]])\
            -> Union[bool, Tuple[bool, str]]:
        """Validate that key_paths is of the form {DataLoadingBlockTypes: (str, str)}."""
        if not isinstance(key_paths, dict):
            return False, f"Field key_paths must be of type dict, instead found {type(key_paths).__name__}"
        for key, value in key_paths.items():
            if key not in [k.value for child in DataLoadingBlockTypes.__subclasses__() for k in child]:
                return False, f"Data loading block key {key} is not a valid key."
            if not isinstance(value, (tuple, list)):
                return False, f"Values for the key_paths dictionary should be tuples or list, " \
                              f"instead found {type(value).__name__}."
            if len(value) != 2:
                return False, f"Values for the key_paths dictionary should have length 2, instead found {len(value)}."
            if not isinstance(value[0], str) or not isinstance(value[1], str):
                return False, f"Key paths should be (str, str) tuples, instead found ({type(value[0]).__name__}, " \
                              f"{type(value[1]).__name__})."
        return True

    @classmethod
    def dlb_default_scheme(cls) -> Dict:
        """The dictionary of default validation rules for a serialized [DataLoadingBlock]."""
        return {
            'loading_block_class': {
                'rules': [str, cls._identifier_validation_hook],
                'required': True,
            },
            'loading_block_module': {
                'rules': [str, cls._identifier_validation_hook],
                'required': True,
            },
            'dlb_id': {
                'rules': [str, cls._serial_id_validation_hook],
                'required': True,
            },
        }

    @classmethod
    def dlp_default_scheme(cls) -> Dict:
        """The dictionary of default validation rules for a serialized [DataLoadingPlan]."""
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


class DataLoadingBlock(ABC):
    """The building blocks of a DataLoadingPlan.

    A [DataLoadingBlock][fedbiomed.common.data._data_loading_plan.DataLoadingBlock] describes an intermediary
    layer between the researcher and the node's filesystem. It allows the node to specify a customization
    in the way data is "perceived" by the data loaders during training.

    A [DataLoadingBlock][fedbiomed.common.data._data_loading_plan.DataLoadingBlock] is identified by its type_id
    attribute. Thus, this attribute should be unique among all
    [DataLoadingBlockTypes][fedbiomed.common.constants.DataLoadingBlockTypes]
    in the same [DataLoadingPlan][fedbiomed.common.data._data_loading_plan.DataLoadingPlan].
    Moreover, we may test equality between a
    [DataLoadingBlock][fedbiomed.common.data._data_loading_plan.DataLoadingBlock]
    and a string by checking its type_id, as a means of easily testing whether a
    [DataLoadingBlock][fedbiomed.common.data._data_loading_plan.DataLoadingBlock] is contained in a collection.

    Correct usage of this class requires creating ad-hoc subclasses.
    The [DataLoadingBlock][fedbiomed.common.data._data_loading_plan.DataLoadingBlock] class is not intended to
    be instantiated directly.

    Subclasses of [DataLoadingBlock][fedbiomed.common.data._data_loading_plan.DataLoadingBlock]
    must respect the following conditions:
    
    1. implement a default constructor
    2. the implemented constructor must call `super().__init__()`
    3. extend the serialize(self) and the deserialize(self, load_from: dict) functions
    4. both serialize and deserialize must call super's serialize and deserialize respectively
    5. the deserialize function must always return self
    6. the serialize function must update the dict returned by super's serialize
    7. implement an apply function that takes arbitrary arguments and applies
            the logic of the loading_block
    8. update the _validation_scheme to define rules for all new fields returned by the serialize function

    Attributes:
        __serialization_id: (str) identifies *one serialized instance* of the DataLoadingBlock
    """

    def __init__(self):
        self.__serialization_id = 'serialized_dlb_' + str(uuid.uuid4())
        self._serialization_validator = SerializationValidation()
        self._serialization_validator.update_validation_scheme(SerializationValidation.dlb_default_scheme())

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
            dlb_id=self.__serialization_id
        )

    def deserialize(self, load_from: dict) -> TDataLoadingBlock:
        """Reconstruct the DataLoadingBlock from a serialized version.

        Args:
            load_from (dict): a dictionary as obtained by the serialize function.
        Returns:
            the self instance
        """
        self._serialization_validator.validate(load_from, FedbiomedLoadingBlockValueError)
        self.__serialization_id = load_from['dlb_id']
        return self

    @abstractmethod
    def apply(self, *args, **kwargs):
        """Abstract method representing an application of the DataLoadingBlock
        """
        pass

    @staticmethod
    def instantiate_class(loading_block: dict) -> TDataLoadingBlock:
        """Instantiate one [DataLoadingBlock][fedbiomed.common.data._data_loading_plan.DataLoadingBlock]
        object of the type defined in the arguments.

        Uses the `loading_block_module` and `loading_block_class` fields of the loading_block argument to
        identify the type of [DataLoadingBlock][fedbiomed.common.data._data_loading_plan.DataLoadingBlock]
        to be instantiated, then calls its default constructor.
        Note that this function **does not call deserialize**.

        Args:
            loading_block (dict): [DataLoadingBlock][fedbiomed.common.data._data_loading_plan.DataLoadingBlock]
                metadata in the format returned by the serialize function.
        Returns:
            A default-constructed instance of a
                [DataLoadingBlock][fedbiomed.common.data._data_loading_plan.DataLoadingBlock]
                of the type defined in the metadata.
        Raises:
           FedbiomedLoadingBlockError: if the instantiation process raised any exception.
        """
        try:
            dlb_module = import_module(loading_block['loading_block_module'])
            dlb = eval(f"dlb_module.{loading_block['loading_block_class']}()")
        except Exception as e:
            msg = f"{ErrorNumbers.FB614.value}: could not instantiate DataLoadingBlock from the following metadata: " +\
                  f"{loading_block} because of {type(e).__name__}: {e}"
            logger.debug(msg)
            raise FedbiomedLoadingBlockError(msg)
        return dlb

    @staticmethod
    def instantiate_key(key_module: str, key_classname: str, loading_block_key_str: str) -> DataLoadingBlockTypes:
        """Imports and loads [DataLoadingBlockTypes][fedbiomed.common.constants.DataLoadingBlockTypes]
        regarding the passed arguments

        Args:
            key_module (str): _description_
            key_classname (str): _description_
            loading_block_key_str (str): _description_

        Raises:
            FedbiomedDataLoadingPlanError: _description_

        Returns:
            DataLoadingBlockTypes: _description_
        """
        try:
            keys = import_module(key_module)
            loading_block_key = eval(f"keys.{key_classname}('{loading_block_key_str}')")
        except Exception as e:
            msg = f"{ErrorNumbers.FB615.value} Error deserializing loading block key " + \
                  f"{loading_block_key_str} with path {key_module}.{key_classname} " + \
                  f"because of {type(e).__name__}: {e}"
            logger.debug(msg)
            raise FedbiomedDataLoadingPlanError(msg)
        return loading_block_key


class MapperBlock(DataLoadingBlock):
    """A [DataLoadingBlock][fedbiomed.common.data._data_loading_plan.DataLoadingBlock] for mapping values.

    This [DataLoadingBlock][fedbiomed.common.data._data_loading_plan.DataLoadingBlock] can be used whenever
    an "indirect mapping" is needed.
    For example, it can be used to implement a correspondence between a set
    of "logical" abstract names and a set of folder names on the filesystem.

    The apply function of this [DataLoadingBlock][fedbiomed.common.data._data_loading_plan.DataLoadingBlock] takes
    a "key" as input (a str) and returns the mapped value corresponding to map[key].
    Note that while the constructor of this class sets a value for type_id,
    developers are recommended to set a more meaningful value that better
    speaks to their application.

    Multiple instances of this loading_block may be used in the same DataLoadingPlan,
    provided that they are given different type_id via the constructor.
    """

    def __init__(self):
        super(MapperBlock, self).__init__()
        self.map = {}
        self._serialization_validator.update_validation_scheme(MapperBlock._extra_validation_scheme())

    def serialize(self) -> dict:
        """Serializes the class in a format similar to json.

        Returns:
            a dictionary of key-value pairs sufficient for reconstructing
            the [DataLoadingBlock][fedbiomed.common.data._data_loading_plan.DataLoadingBlock].
        """
        ret = super(MapperBlock, self).serialize()
        ret.update({'map': self.map})
        return ret

    def deserialize(self, load_from: dict) -> DataLoadingBlock:
        """Reconstruct the [DataLoadingBlock][fedbiomed.common.data._data_loading_plan.DataLoadingBlock]
        from a serialized version.

        Args:
            load_from (dict): a dictionary as obtained by the serialize function.
        Returns:
            the self instance
        """
        super(MapperBlock, self).deserialize(load_from)
        self.map = load_from['map']
        return self

    def apply(self, key):
        """Returns the value mapped to the key, if it exists.

        Raises:
            FedbiomedLoadingBlockError: if map is not a dict or the key does not exist.
        """
        if not isinstance(self.map, dict) or key not in self.map:
            msg = f"{ErrorNumbers.FB614.value} Mapper block error: no key '{key}' in mapping dictionary"
            logger.debug(msg)
            raise FedbiomedLoadingBlockError(msg)
        return self.map[key]

    @classmethod
    def _extra_validation_scheme(cls):
        return {
            'map': {
                'rules': [dict],
                'required': True
            }
        }


class DataLoadingPlan(Dict[DataLoadingBlockTypes, DataLoadingBlock]):
    """Customizations to the way the data is loaded and presented for training.

    A DataLoadingPlan is a dictionary of {name: DataLoadingBlock} pairs. Each
    [DataLoadingBlock][fedbiomed.common.data._data_loading_plan.DataLoadingBlock]
    represents a customization to the way data is loaded and presented to the researcher.
    These customizations are defined by the node, but they operate on a Dataset class,
    which is defined by the library and instantiated by the researcher.

    To exploit this functionality, a Dataset must be modified to accept the
    customizations provided by the DataLoadingPlan. To simplify this process,
    we provide the [DataLoadingPlanMixin][fedbiomed.common.data._data_loading_plan.DataLoadingPlanMixin] class below.

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
        self.dlp_id = 'dlp_' + str(uuid.uuid4())
        self.desc = ""
        self.target_dataset_type = DatasetTypes.NONE
        self._serialization_validation = SerializationValidation()
        self._serialization_validation.update_validation_scheme(SerializationValidation.dlp_default_scheme())

    def __setitem__(self, key: DataLoadingBlockTypes, value: DataLoadingBlock):
        """Type-check the arguments then call dict.__setitem__."""
        if not isinstance(key, DataLoadingBlockTypes):
            msg = f"{ErrorNumbers.FB615.value} Key {key} is not of enum type DataLoadingBlockTypes in" + \
                  f" DataLoadingPlan {self}"
            logger.debug(msg)
            raise FedbiomedDataLoadingPlanValueError(msg)
        if not isinstance(value, DataLoadingBlock):
            msg = f"{ErrorNumbers.FB615.value} Value {value} is not of type DataLoadingBlock in" + \
                  f" DataLoadingPlan {self}"
            logger.debug(msg)
            raise FedbiomedDataLoadingPlanValueError(msg)
        super().__setitem__(key, value)

    def serialize(self) -> Tuple[dict, List]:
        """Serializes the class in a format similar to json.

        Returns:
            a tuple sufficient for reconstructing the DataLoading plan. It includes:
                - a dictionary of key-value pairs with the
                [DataLoadingPlan][fedbiomed.common.data._data_loading_plan.DataLoadingPlan] parameters.
                - a list of dict containing the data for reconstruction all the DataLoadingBlock
                    of the [DataLoadingPlan][fedbiomed.common.data._data_loading_plan.DataLoadingPlan] 
        """
        return dict(
            dlp_id=self.dlp_id,
            dlp_name=self.desc,
            target_dataset_type=self.target_dataset_type.value,
            loading_blocks={key.value: dlb.get_serialization_id() for key, dlb in self.items()},
            key_paths={key.value: (f"{key.__module__}", f"{key.__class__.__qualname__}") for key in self.keys()}
        ), [dlb.serialize() for dlb in self.values()]

    def deserialize(self, serialized_dlp: dict, serialized_loading_blocks: List[dict]) -> TDataLoadingPlan:
        """Reconstruct the DataLoadingPlan][fedbiomed.common.data._data_loading_plan.DataLoadingPlan] from a serialized version.

        !!! warning "Calling this function will *clear* the contained [DataLoadingBlockTypes]."
            This function may not be used to "update" nor to "append to"
            a [DataLoadingPlan][fedbiomed.common.data._data_loading_plan.DataLoadingPlan].

        Args:
            serialized_dlp: a dictionary of data loading plan metadata, as obtained from the first output of the
                serialize function
            serialized_loading_blocks: a list of dictionaries of loading_block metadata, as obtained from the
                second output of the serialize function
        Returns:
            the self instance
        """
        self._serialization_validation.validate(serialized_dlp, FedbiomedDataLoadingPlanValueError)

        self.clear()
        self.dlp_id = serialized_dlp['dlp_id']
        self.desc = serialized_dlp['dlp_name']
        self.target_dataset_type = DatasetTypes(serialized_dlp['target_dataset_type'])
        for loading_block_key_str, dlb_id in serialized_dlp['loading_blocks'].items():
            key_module, key_classname = serialized_dlp['key_paths'][loading_block_key_str]
            loading_block_key = DataLoadingBlock.instantiate_key(key_module, key_classname, loading_block_key_str)
            loading_block = next(filter(lambda x: x['dlb_id'] == dlb_id,
                                        serialized_loading_blocks))
            dlb = DataLoadingBlock.instantiate_class(loading_block)
            self[loading_block_key] = dlb.deserialize(loading_block)
        return self

    def __str__(self):
        """User-friendly string representation"""
        return f"Data Loading Plan {self.desc} id: {self.dlp_id} " \
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
            a DatasetTypes enum element which identifies the type of the dataset.
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
        msg = f"{ErrorNumbers.FB615.value} Trying to infer dataset type of {dataset} is not supported " + \
            f"for datasets of type {dataset.__class__.__qualname__}"
        logger.debug(msg)
        raise FedbiomedDataLoadingPlanValueError(msg)


class DataLoadingPlanMixin:
    """Utility class to enable DLP functionality in a dataset.

    Any Dataset class that inherits from [DataLoadingPlanMixin] will have the
    basic tools necessary to support a [DataLoadingPlan][fedbiomed.common.data._data_loading_plan.DataLoadingPlan].
    Typically, the logic of each specific DataLoadingBlock in the [DataLoadingPlan][fedbiomed.common.data._data_loading_plan.DataLoadingPlan]
    will be implemented in the form of hooks that are called within the Dataset's implementation
    using the helper function apply_dlb defined below.
    """

    def __init__(self):
        self._dlp = None

    def set_dlp(self, dlp: DataLoadingPlan):
        """Sets the dlp if the target dataset type is appropriate"""
        if not isinstance(dlp, DataLoadingPlan):
            msg = f"{ErrorNumbers.FB615.value} Trying to set a DataLoadingPlan but the argument is of type " + \
                  f"{type(dlp).__name__}"
            logger.debug(msg)
            raise FedbiomedDataLoadingPlanValueError(msg)

        dataset_type = DataLoadingPlan.infer_dataset_type(self)  # `self` here will refer to the Dataset instance
        if dlp.target_dataset_type != DatasetTypes.NONE and dataset_type != dlp.target_dataset_type:
            raise FedbiomedDataLoadingPlanValueError(f"Trying to set {dlp} on dataset of type {dataset_type.value} but "
                                                     f"the target type is {dlp.target_dataset_type}")
        elif dlp.target_dataset_type == DatasetTypes.NONE:
            dlp.target_dataset_type = dataset_type
        self._dlp = dlp

    def clear_dlp(self):
        self._dlp = None

    def apply_dlb(self, default_ret_value: Any, dlb_key: DataLoadingBlockTypes,
                  *args: Optional[Any], **kwargs: Optional[Any]) -> Any:
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
            *args: forwarded to the DataLoadingBlock's apply function
            **kwargs: forwarded to the DataLoadingBlock's apply function
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
