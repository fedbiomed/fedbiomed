# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Secure Aggregation setup on the node"""
from typing import List
from abc import ABC, abstractmethod
from enum import Enum
import time
import random

from fedbiomed.common.constants import ErrorNumbers, SecaggElementTypes
from fedbiomed.common.exceptions import FedbiomedSecaggError
from fedbiomed.common.message import NodeMessages, SecaggReply
from fedbiomed.common.logger import logger
from fedbiomed.common.validator import Validator, ValidatorError

from fedbiomed.node.environ import environ
from fedbiomed.node.secagg_manager import SecaggServkeyManager, SecaggBiprimeManager


class SecaggSetup(ABC):
    """
    Sets up a Secure Aggregation context element on the node side.
    """

    def __init__(
            self,
            researcher_id: str,
            secagg_id: str,
            job_id: str,
            sequence: int,
            parties: List[str]):
        """Constructor of the class.

        Args:
            researcher_id: ID of the researcher that requests setup
            secagg_id: ID of secagg context element for this setup request
            job_id: ID of the job to which this secagg context element is attached (empty string if no attached job)
            sequence: unique sequence number of setup request
            parties: List of parties participating to the secagg context element setup

        Raises:
            FedbiomedSecaggError: bad argument type or value
        """
        # check arguments
        self._v = Validator()
        for param, type in [(researcher_id, str), (secagg_id, str), (job_id, str), (sequence, int)]:
            try:
                self._v.validate(param, type)
            except ValidatorError as e:
                errmess = f'{ErrorNumbers.FB318.value}: bad parameter `{param}` should be a {type}: {e}'
                logger.error(errmess)
                raise FedbiomedSecaggError(errmess)
        for param, name in [(researcher_id, 'researcher_id'), (secagg_id, 'secagg_id')]:
            if not param:
                errmess = f'{ErrorNumbers.FB318.value}: bad parameter `{name}` should not be empty string'
                logger.error(errmess)
                raise FedbiomedSecaggError(errmess)
        try:
            self._v.validate(parties, list)
            for p in parties:
                self._v.validate(p, str)
        except ValidatorError as e:
            errmess = f'{ErrorNumbers.FB318.value}: bad parameter `parties` must be a list of strings: {e}'
            logger.error(errmess)
            raise FedbiomedSecaggError(errmess)
        if len(parties) < 3:
            errmess = f'{ErrorNumbers.FB318.value}: bad parameter `parties` : {parties} : need  ' \
                'at least 3 parties for secure aggregation'
            logger.error(errmess)
            raise FedbiomedSecaggError(errmess)
        if researcher_id != parties[0]:
            errmess = f'{ErrorNumbers.FB318.value}: bad parameter `researcher_id` : {researcher_id} : ' \
                'needs to be the same as the first secagg party'
            logger.error(errmess)
            raise FedbiomedSecaggError(errmess)

        # assign argument values
        self._researcher_id = researcher_id
        self._secagg_id = secagg_id
        self._job_id = job_id
        self._sequence = sequence
        self._parties = parties

    def researcher_id(self) -> str:
        """Getter for `researcher_id`

        Returns:
            researcher unique ID
        """
        return self._researcher_id

    def secagg_id(self) -> str:
        """Getter for `secagg_id`

        Returns:
            secagg context element unique ID
        """
        return self._secagg_id

    def job_id(self) -> str:
        """Getter for `job_id`

        Returns:
            ID of the job to which this secagg context element is attached (empty string if no attached job)
        """
        return self._job_id

    def sequence(self) -> str:
        """ Getter for `sequence`

        Returns:
            sequence number for this request
        """
        return self._sequence

    @abstractmethod
    def element(self) -> Enum:
        """Getter for secagg context element type

        Returns:
            secagg context element name
        """

    def _create_secagg_reply(self, message: str = '', success: bool = False) -> SecaggReply:
        """Create reply message for researcher after secagg setup phase.

        Args:
            message: text information concerning the secagg setup
            success: `True` if secagg element setup was successful, `False` otherway

        Returns:
            message to return to the researcher
        """

        # If round is not successful log error message
        if not success:
            logger.error(message)

        return NodeMessages.reply_create(
            {
                'researcher_id': self._researcher_id,
                'secagg_id': self._secagg_id,
                'sequence': self._sequence,
                'success': success,
                'node_id': environ['NODE_ID'],
                'msg': message,
                'command': 'secagg'
            }
        ).get_dict()

    @abstractmethod
    def setup(self) -> SecaggReply:
        """Set up a secagg context element.

        Returns:
            message to return to the researcher after the setup
        """


class SecaggServkeySetup(SecaggSetup):
    """
    Sets up a server key Secure Aggregation context element on the node side.
    """
    def __init__(
            self,
            researcher_id: str,
            secagg_id: str,
            job_id: str,
            sequence: int,
            parties: List[str]):
        """Constructor of the class.

        Args:
            researcher_id: ID of the researcher that requests setup
            secagg_id: ID of secagg context element for this setup request
            job_id: ID of the job to which this secagg context element is attached
            sequence: unique sequence number of setup request
            parties: List of parties participating to the secagg context element setup

        Raises:
            FedbiomedSecaggError: bad argument type or value 
        """
        super().__init__(researcher_id, secagg_id, job_id, sequence, parties)

        if not self._job_id:
            errmess = f'{ErrorNumbers.FB318.value}: bad parameter `job_id` must be a non empty string'
            logger.error(errmess)
            raise FedbiomedSecaggError(errmess)


    def element(self) -> Enum:
        """Getter for secagg context element type

        Returns:
            secagg context element name
        """
        return SecaggElementTypes.SERVER_KEY

    def setup(self) -> SecaggReply:
        """Set up the server key secagg context element.

        Returns:
            message to return to the researcher after the setup
        """
        manager = SecaggServkeyManager()
        # also checks that `context` is attached to the job `self._job_id`
        context = manager.get(self._secagg_id, self._job_id)

        if context is None:
            # create a context if it does not exist yet
            time.sleep(4)
            servkey_share = str(random.randrange(10**6))
            logger.info("Not implemented yet, PUT SECAGG SERVKEY GENERATION PAYLOAD HERE, "
                        f"secagg_id='{self._secagg_id}'")

            manager.add(self._secagg_id, self._parties, self._job_id, servkey_share)

        logger.info(f"Completed secagg servkey setup for node_id='{environ['NODE_ID']}' secagg_id='{self._secagg_id}'")
        msg = self._create_secagg_reply('', True)
        return msg


class SecaggBiprimeSetup(SecaggSetup):
    """
    Sets up a biprime Secure Aggregation context element on the node side.
    """
    def __init__(
            self,
            researcher_id: str,
            secagg_id: str,
            job_id: str,
            sequence: int,
            parties: List[str]):
        """Constructor of the class.

        Args:
            researcher_id: ID of the researcher that requests setup
            secagg_id: ID of secagg context element for this setup request
            job_id: must be an empty string for a biprime context element (not attached to a job)
            sequence: unique sequence number of setup request
            parties: List of parties participating to the secagg context element setup

        Raises:
            FedbiomedSecaggError: bad argument type or value
        """
        super().__init__(researcher_id, secagg_id, job_id, sequence, parties)

        if self._job_id:
            errmess = f'{ErrorNumbers.FB318.value}: bad parameter `job_id` must be an empty string'
            logger.error(errmess)
            raise FedbiomedSecaggError(errmess)

    def element(self) -> Enum:
        """Getter for secagg context element type

        Returns:
            secagg context element name
        """
        return SecaggElementTypes.BIPRIME

    def setup(self) -> SecaggReply:
        """Set up the biprime secagg context element.

        Returns:
            message to return to the researcher after the setup
        """
        manager = SecaggBiprimeManager()
        context = manager.get(self._secagg_id)

        if context is None:
            # create a context if it does not exist yet
            time.sleep(6)
            biprime = str(random.randrange(10**12))
            logger.info("Not implemented yet, PUT SECAGG BIPRIME GENERATION PAYLOAD HERE, "
                        f"secagg_id='{self._secagg_id}'")

            manager.add(self._secagg_id, self._parties, biprime)

        logger.info(f"Completed secagg biprime setup for node_id='{environ['NODE_ID']}' secagg_id='{self._secagg_id}'")
        msg = self._create_secagg_reply('', True)
        return msg
