# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Union, Callable, Dict



class FedbiomedDataset:

    _framework_transform : Optional[Union[Callable, Dict[str, Callable]]] = None

    def __init__(self, framework_transform : Optional[Union[Callable, Dict[str, Callable]]] = None):
        if framework_transform:
            # TODO: check type
            self._framework_transform = framework_transform

    def framework_transform(self) -> Optional[Union[Callable, Dict[str, Callable]]]:
        return self._framework_transform
