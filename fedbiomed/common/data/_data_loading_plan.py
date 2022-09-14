import uuid
from typing import Any, Dict, List, Tuple, TypeVar, Optional
from abc import ABC, abstractmethod

from fedbiomed.common.constants import DataLoadingBlockTypes
from fedbiomed.common.exceptions import FedbiomedLoadingBlockError


TDataLoadingPlan = TypeVar("TDataLoadingPlan", bound="DataLoadingPlan")
TDataLoadingBlock = TypeVar("TDataLoadingBlock", bound="DataLoadingBlock")


class DataLoadingBlock(ABC):
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
        super(DataLoadingBlock, self).__init__()
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
        self.__serialization_id = load_from['loading_block_serialization_id']
        return self

    @abstractmethod
    def apply(self, *args, **kwargs):
        """Abstract method representing an application of the DataLoadingBlock
        """
        pass


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


class DataLoadingPlan(Dict[DataLoadingBlockTypes, DataLoadingBlock]):
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
    """
    def __init__(self, *args, **kwargs):
        super(DataLoadingPlan, self).__init__(*args, **kwargs)
        self.dlp_id = 'dlp_' + str(uuid.uuid4())
        self.desc = ""

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
            loading_blocks={key.value: dlb.get_serialization_id() for key, dlb in self.items()},
            key_paths={key.value: (f"{key.__module__}", f"{key.__class__.__qualname__}") for key in self.keys()}
        ), [dlb.serialize() for dlb in self.values()]

    def deserialize(self, serialized_dlp: dict, serialized_loading_blocks: List[dict]) -> TDataLoadingPlan:
        """Reconstruct the DataLoadingPlan from a serialized version.

        :warning: Calling this function will *clear* the contained
            DataLoadingBlockTypes. This function may not be used to "update"
            nor to "append to" a DataLoadingPlan.

        Args:
            serialized_dlp: a dictionary of data loading plan metadata, as obtained from the first output of the
                serialize function
            serialized_loading_blocks: a list of dictionaries of loading_block metadata, as obtained from the second output
                of the serialize function
        Returns:
            the self instance
        """
        self.clear()
        self.dlp_id = serialized_dlp['dlp_id']
        self.desc = serialized_dlp['dlp_name']
        for loading_block_key_str, loading_block_serialization_id in serialized_dlp['loading_blocks'].items():
            loading_block = next(filter(lambda x: x['loading_block_serialization_id'] == loading_block_serialization_id,
                                        serialized_loading_blocks))
            exec(f"import {loading_block['loading_block_module']}")
            dlb = eval(f"{loading_block['loading_block_module']}.{loading_block['loading_block_class']}()")
            key_module, key_classname = serialized_dlp['key_paths'][loading_block_key_str]
            exec(f"import {key_module}")
            loading_block_key = eval(f"{key_module}.{key_classname}('{loading_block_key_str}')")
            self[loading_block_key] = dlb.deserialize(loading_block)
        return self

    def deserialize_loading_blocks_from_mapping(self,
                                                serialized_loading_blocks_mapping: Dict[str, dict]) -> TDataLoadingPlan:
        """Reconstruct only the loading blocks of the Data Loading Plan

        The input argument must be of the form {str: dict} where the str key is a name that will identify each loading
        block, and the dict value is a dictionary of loading block metadata used to deserialize the loading block.

        This function may be used to update an existing DataLoadingPlan by adding the deserialized loading blocks.

        Args:
            serialized_loading_blocks_mapping : a dictionary of {name: metadata} pairs
        Returns:
            the self instance
        """

        for loading_block_key, loading_block in serialized_loading_blocks_mapping.items():
            exec(f"import {loading_block['loading_block_module']}")
            dlb = eval(f"{loading_block['loading_block_module']}.{loading_block['loading_block_class']}()")
            self[loading_block_key] = dlb.deserialize(loading_block)
        return self

    def __str__(self):
        """User-friendly string representation"""
        return f"Data Loading Plan {self.desc} id: {self.dlp_id} "\
               f"containing: {'; '.join([k.value for k in self.keys()])}"


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

    def set_dlp(self, dlp):
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
        if self._dlp is not None and dlb_key in self._dlp:
            return self._dlp[dlb_key].apply(*args, **kwargs)
        else:
            return default_ret_value





