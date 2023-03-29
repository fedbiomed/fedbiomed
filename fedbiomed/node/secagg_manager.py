# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Interface with the node secure aggregation element database
"""

from fedbiomed.common.constants import ErrorNumbers, SecaggElementTypes
from fedbiomed.common.exceptions import FedbiomedSecaggError
from fedbiomed.common.logger import logger

from fedbiomed.node.environ import environ
from fedbiomed.common.secagg_manager import SecaggServkeyManager, \
    SecaggBiprimeManager, \
    BaseSecaggManager

# Instantiate one manager for each secagg element type
SKManager = SecaggServkeyManager(environ['DB_PATH'])
BPrimeManager = SecaggBiprimeManager(environ['DB_PATH'])


class SecaggManager:
    """Wrapper class for returning any type of node secagg element database manager
    """

    element2class = {
        SecaggElementTypes.SERVER_KEY.name: SKManager,
        SecaggElementTypes.BIPRIME.name: BPrimeManager
    }

    def __init__(self, element: int):
        """Constructor of the class
        """
        self._element = element

    def __call__(self) -> BaseSecaggManager:
        """Return a node secagg element database manager object.

        Returns:
            an existing secagg element database manager object
        """

        if self._element in [m.value for m in SecaggElementTypes]:
            element = SecaggElementTypes(self._element)
        else:
            error_msg = f'{ErrorNumbers.FB318.value}: received bad message: ' \
                        f'incorrect `element` {self._element}'
            logger.error(error_msg)
            raise FedbiomedSecaggError(error_msg)

        try:
            return SecaggManager.element2class[element.name]
        except Exception as e:
            raise FedbiomedSecaggError(
                f'{ErrorNumbers.FB318.value}: Missing secure aggregation component for this element type: Error{e}'
            )
