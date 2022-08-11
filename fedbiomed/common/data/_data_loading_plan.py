import uuid
from typing import Any, Dict, List, Tuple, TypeVar
from abc import ABC, abstractmethod

from fedbiomed.common.constants import DataLoadingPipelineKeys


TDataLoadingPlan = TypeVar("TDataLoadingPlan", bound="DataLoadingPlan")
TDataPipeline = TypeVar("TDataPipeline", bound="DataPipeline")


class DataPipeline(ABC):
    """The building blocks of a DataLoadingPlan.

    A DataPipeline describes an intermediary layer between the researcher
    and the node's filesystem. It allows the node to specify a customization
    in the way data is "perceived" by the data loaders during training.

    A DataPipeline is identified by its type_id attribute. Thus this
    attribute should be unique among all DataPipelines in the same
    DataLoadingPlan. Moreover, we may test equality between a
    DataPipeline and a string by checking its type_id, as a means of
    easily testing whether a DataPipeline is contained in a collection.

    Correct usage of this class requires creating ad-hoc subclasses.
    The DataPipeline class is not intended to be instantiated directly.

    Subclasses of DataPipline must respect the following conditions:
    1. implement a constructor taking exactly one argument, a type_id string
    2. the implemented constructor must call super().__init__(type_id)
    3. extend the serialize(self) and the deserialize(self, load_from: dict) functions
    4. both serialize and deserialize must call super's serialize and deserialize respectively
    5. the deserialize function must always return self
    6. the serialize function must update the dict returned by super's serialize
    7. implement an apply function that takes arbitrary arguments and applies
       the logic of the pipeline

    Attributes:
        __serialization_id: (str) identifies *one serialized instance* of the DataPipeline
    """
    def __init__(self):
        super(DataPipeline, self).__init__()
        self.__serialization_id = 'serialized_dp_' + str(uuid.uuid4())

    def get_serialization_id(self):
        """Expose serialization id as read-only"""
        return self.__serialization_id

    def serialize(self) -> dict:
        """Serializes the class in a format similar to json.

        Returns:
             a dictionary of key-value pairs sufficient for reconstructing
             the DataPipeline.
        """
        return dict(
            pipeline_class=self.__class__.__qualname__,
            pipeline_module=self.__module__,
            pipeline_serialization_id=self.__serialization_id
        )

    def deserialize(self, load_from: dict) -> TDataPipeline:
        """Reconstruct the DataPipeline from a serialized version.

        Args:
            load_from: a dictionary as obtained by the serialize function.
        Returns:
            the self instance
        """
        self.__serialization_id = load_from['pipeline_serialization_id']
        return self

    @abstractmethod
    def apply(self, *args, **kwargs):
        """Abstract method representing an application of the DataPipeline
        """
        pass


class MapperDP(DataPipeline):
    """A DataPipeline for mapping values.

    This Datapipeline can be used whenever an "indirect mapping" is needed.
    For example, it can be used to implement a correspondence between a set
    of "logical" abstract names and a set of folder names on the filesystem.

    The apply function of this DataPipeline takes a "key" as input (a str)
    and returns the mapped value corresponding to map[key].
    Note that while the constructor of this class sets a value for type_id,
    developers are recommended to set a more meaningful value that better
    speaks to their application.

    Multiple instances of this pipeline may be used in the same DataLoadingPlan,
    provided that they are given different type_id via the constructor.
    """
    def __init__(self):
        super(MapperDP, self).__init__()
        self.map = {}

    def serialize(self) -> dict:
        """Serializes the class in a format similar to json.

        Returns:
             a dictionary of key-value pairs sufficient for reconstructing
             the DataPipeline.
        """
        ret = super(MapperDP, self).serialize()
        ret.update({'map': self.map})
        return ret

    def deserialize(self, load_from: dict) -> DataPipeline:
        """Reconstruct the DataPipeline from a serialized version.

        Args:
            load_from: a dictionary as obtained by the serialize function.
        Returns:
            the self instance
        """
        super(MapperDP, self).deserialize(load_from)
        self.map = load_from['map']
        return self

    def apply(self, key):
        return self.map[key]


class DataLoadingPlan(Dict[DataLoadingPipelineKeys, DataPipeline]):
    """Customizations to the way the data is loaded and presented for training.

    A DataLoadingPlan is a dictionary of {name: DataPipeline} pairs. Each
    DataPipeline represents a customization to the way data is loaded and
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
             a dictionary of key-value pairs sufficient for reconstructing
             the DataLoadingPlan.
        """
        return dict(
            dlp_id=self.dlp_id,
            dlp_name=self.desc,
            pipelines={key.value: dp.get_serialization_id() for key, dp in self.items()},
            key_paths={key.value: (f"{key.__module__}", f"{key.__class__.__qualname__}") for key in self.keys()}
        ), [dp.serialize() for dp in self.values()]

    def deserialize(self, serialized_dlp: dict, serialized_pipelines: List[dict]) -> TDataLoadingPlan:
        """Reconstruct the DataLoadingPlan from a serialized version.

        The format of the input argument is expected to be an 'aggregated serialized' version, as defined by the output
        of the 'DataLoadingPlan.aggregate_serialized_metadata` function.

        :warning: Calling this function will *clear* the contained
            DataPipelines. This function may not be used to "update"
            nor to "append to" a DataLoadingPlan.

        Args:
            serialized_dlp: a dictionary of data loading plan metadata, as obtained from the first output of the
                            serialize function
            serialized_pipelines: a list of dictionaries of pipeline metadata, as obtained from the second output of
                                  the serialize function
        Returns:
            the self instance
        """
        self.clear()
        self.dlp_id = serialized_dlp['dlp_id']
        self.desc = serialized_dlp['dlp_name']
        for pipeline_key_str, pipeline_serialization_id in serialized_dlp['pipelines'].items():
            pipeline = next(filter(lambda x: x['pipeline_serialization_id'] == pipeline_serialization_id,
                                   serialized_pipelines))
            exec(f"import {pipeline['pipeline_module']}")
            dp = eval(f"{pipeline['pipeline_module']}.{pipeline['pipeline_class']}()")
            key_module, key_classname = serialized_dlp['key_paths'][pipeline_key_str]
            exec(f"import {key_module}")
            pipeline_key = eval(f"{key_module}.{key_classname}('{pipeline_key_str}')")
            self[pipeline_key] = dp.deserialize(pipeline)
        return self

    def __str__(self):
        """User-friendly string representation"""
        return f"Data Loading Plan {self.desc} id: {self.dlp_id} "\
               f"containing: {'; '.join([k.value for k in self.keys()])}"


class DataLoadingPlanMixin:
    """Utility class to enable DLP functionality in a dataset.

    Any Dataset class that inherits from DataLoadingPlanMixin will have the
    basic tools necessary to support a DataLoadingPlan. Typically, the logic
    of each specific DataPipeline in the DataLoadingPlan will be implemented
    in the form of hooks that are called within the Dataset's implementation
    using the helper function apply_dp defined below.
    """
    def __init__(self):
        self._dlp = None

    def set_dlp(self, dlp):
        self._dlp = dlp

    def clear_dlp(self):
        self._dlp = None

    def apply_dp(self, default_ret_value: Any, dp_key: str, *args, **kwargs):
        """Apply one DataPipeline identified by its key.

        Note that we want to easily support the case where the DataLoadingPlan
        is not activated, or the requested pipeline is not contained in the
        DataLoadingPlan. This is achieved by providing a default return value
        to be returned when the above conditions are met. Hence, most of the
        calls to apply_dp will look like this:
        ```
        value = self.apply_dp(value, 'my-pipeline', my_pipeline_args)
        ```
        This will ensure that value is not changed if the DataLoadingPlan is
        not active.

        Args:
            default_ret_value: the value to be returned in case that the dlp
            functionality is not required
            dp_key: the key of the DataPipeline to be applied
            args: forwarded to the DataPipeline's apply function
            kwargs: forwarded to the DataPipeline's apply function
        Returns:
             the output of the DataPipeline's apply function, or
             the default_ret_value when dlp is None or it does not contain
             the requested pipeline
        """
        if self._dlp is not None and dp_key in self._dlp:
            return self._dlp[dp_key].apply(*args, **kwargs)
        else:
            return default_ret_value





