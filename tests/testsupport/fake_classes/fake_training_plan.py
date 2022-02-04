# this fake class is stored outside unit tests in order to avoid re-wirting 
# the same fake classes each time we are desinging a  unit test

from typing import Dict, Any, List
import time


## Fakes TrainingPlan (either `fedbiomed.common.torchnn`` or `fedbiomed.common.fedbiosklearn`)
class FakeModel:
    # Fake model that mimics a Training Plan model
    SLEEPING_TIME = 1 # time that simulate training (in seconds)
    def __init__(self, *args, **kwargs):
        pass
    def load(self, path:str, to_params:bool):
        pass
    def save(self, filename:str, results: Dict[str, Any]):
        pass
    def set_dataset(self, path:str):
        pass
    def training_routine(self, **kwargs):
        time.sleep(FakeModel.SLEEPING_TIME)
    def after_training_params(self)-> List[int]:
        return [1, 2, 3, 4]