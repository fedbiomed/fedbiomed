from enum import Enum
from fedbiomed.common.data import DataPipeline
from fedbiomed.common.constants import DataLoadingPipelineKeys


class PipelineKeysForTesting(DataLoadingPipelineKeys, Enum):
    PIPELINE_FOR_TESTING: str = 'pipeline_for_testing'
    OTHER_TESTING_PIPELINE: str = 'other_testing_pipeline'
    TESTING_MAPPER: str = 'testing_mapper'
    MODIFY_GETITEM: str = 'modify_getitem'


class PipelineForTesting(DataPipeline):
    def __init__(self):
        super(PipelineForTesting, self).__init__()
        self.data = {'my': 'data'}

    def serialize(self):
        ret = super(PipelineForTesting, self).serialize()
        ret.update({'data': self.data})
        return ret

    def deserialize(self, load_from):
        super(PipelineForTesting, self).deserialize(load_from)
        self.data = load_from['data']
        return self

    def apply(self):
        return self.data.values()


class ModifyGetItemDP(DataPipeline):
    def __init__(self):
        super(ModifyGetItemDP, self).__init__()
        self.type_id = 'modify-getitem'

    def apply(self):
        return 'modified-value'

