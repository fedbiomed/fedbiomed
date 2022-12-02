from enum import Enum
from fedbiomed.common.data import DataLoadingBlock
from fedbiomed.common.constants import DataLoadingBlockTypes


class LoadingBlockTypesForTesting(DataLoadingBlockTypes, Enum):
    LOADING_BLOCK_FOR_TESTING: str = 'loading_block_for_testing'
    OTHER_LOADING_BLOCK_FOR_TESTING: str = 'other_loading_block_for_testing'
    TESTING_MAPPER: str = 'testing_mapper'
    MODIFY_GETITEM: str = 'modify_getitem'


class LoadingBlockForTesting(DataLoadingBlock):
    def __init__(self):
        super(LoadingBlockForTesting, self).__init__()
        self.data = {'my': 'data'}
        self._serialization_validator.update_validation_scheme({
            'data': {
                'rules': [dict],
                'required': True
            }
        })

    def serialize(self):
        ret = super(LoadingBlockForTesting, self).serialize()
        ret.update({'data': self.data})
        return ret

    def deserialize(self, load_from):
        super(LoadingBlockForTesting, self).deserialize(load_from)
        self.data = load_from['data']
        return self

    def apply(self):
        return self.data.values()


class ModifyGetItemDP(DataLoadingBlock):
    def __init__(self):
        super(ModifyGetItemDP, self).__init__()
        self.type_id = 'modify-getitem'

    def apply(self):
        return 'modified-value'


# class for cheating the ABC into running the abstract methods
class TestAbstractsBlock(DataLoadingBlock):
    def apply(self, *args, **kwargs):
        super(TestAbstractsBlock, self).apply(*args, *kwargs)
