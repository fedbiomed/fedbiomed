from fedbiomed.common.data.data_loading_plan import DataPipeline


class PipelineForTesting(DataPipeline):
    def __init__(self):
        self.type_id = 'pipeline_for_testing'
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
    def __init__(self):
        super().__init__()
        self.type_id = 'modify-getitem'

    def apply(self):
        return 'modified-value'

