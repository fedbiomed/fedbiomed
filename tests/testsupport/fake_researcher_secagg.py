from typing import Union, List
from unittest.mock import MagicMock

from fedbiomed.common.constants import SecureAggregationSchemes
from fedbiomed.researcher.secagg._secure_aggregation import _SecureAggregation, SecureAggregation

FAKE_CONTEXT_VALUE = "MY_CONTEXT"


class FakeSecaggContext:
    def __init__(self, parties: List[str], experiment_id: str):
        self.parties = parties
        self.experiment_id = experiment_id
        self.stat = False
        self.cont = None
        self.success = True

    @property
    def status(self) -> bool:
        return self.stat

    @property
    def context(self) -> Union[dict, None]:
        return self.cont

    def setup(self, timeout: float = 0) -> bool:
        self.stat = self.success
        if self.stat:
            self.cont = FAKE_CONTEXT_VALUE
        else:
            self.cont = None
        return self.stat

    def set_setup_success(self, success: bool = True):
        """Choose whether next `setup`s will fail or succeed.

        This does not exist in real `SecaggContext` and is added for mocking purpose.
        """
        self.success = success

    def load_state_breakpoint(self, *arg, **kwargs):
        pass


class FakeSecaggServkeyContext(FakeSecaggContext):
    pass


class FakeSecaggBiprimeContext(FakeSecaggContext):
    def __init__(self, parties: List[str]):
        super().__init__(parties, '')


class FakeSecAgg(SecureAggregation):
    arg_train_arguments = None
    def __init__(self, *args, scheme: SecureAggregationSchemes = SecureAggregationSchemes.LOM, **kwargs) -> None:
        self.__secagg = MagicMock(spec=_SecureAggregation,
                                  train_arguments = MagicMock(return_value=FakeSecAgg.arg_train_arguments),
                                  **kwargs)

    def __getattr__(self, item: str):
        return getattr(self.__secagg, item)
