from fedbiomed.common.data import DataPipeline


class PipelineForTesting(DataPipeline):
    def __init__(self, type_id: str):
        super(PipelineForTesting, self).__init__(type_id)
        self.data = {'my': 'data'}

    def serialize(self):
        ret = super(PipelineForTesting, self).serialize()
        ret.update({'data': self.data})
        return ret

    def load(self, load_from):
        super(PipelineForTesting, self).load(load_from)
        self.data = load_from['data']
        return self

    def apply(self):
        return self.data.values()


class ModifyGetItemDP(DataPipeline):
    def __init__(self, type_id: str):
        super(ModifyGetItemDP, self).__init__(type_id)
        self.type_id = 'modify-getitem'

    def apply(self):
        return 'modified-value'

