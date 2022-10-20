"""Secure Aggregation setup on the node"""
from typing import List
from abc import ABC, abstractmethod

from fedbiomed.common.constants import SecaggElementTypes
from fedbiomed.common.message import NodeMessages, SecaggReply
from fedbiomed.common.logger import logger

from fedbiomed.node.environ import environ


class SecaggSetup(ABC):
    """
    Sets up a Secure Aggregation context element on the node side.
    """

    def __init__(
            self,
            researcher_id: str,
            secagg_id: str,
            sequence: int,
            parties: List[str]):
        """Constructor of the class.

        Args:
            researcher_id: ID of the researcher that requests setup
            secagg_id: ID of secagg context element for this setup request
            sequence: unique sequence number of setup request
            parties: List of parties participating to the secagg context element setup
        """
        # we can suppose input was properly checked before instantiating this object

        self._researcher_id = researcher_id
        self._secagg_id = secagg_id
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

    def sequence(self) -> str:
        """ Getter for `sequence`

        Returns:
            sequence number for this request
        """
        return self._sequence

    @abstractmethod
    def element(self) -> str:
        """Getter for secagg context element type

        Returns:
            secagg context element name
        """
        pass

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
        pass


class SecaggServkeySetup(SecaggSetup):
    """
    Sets up a server key Secure Aggregation context element on the node side.
    """
    def __init__(
            self,
            researcher_id: str,
            secagg_id: str,
            sequence: int,
            parties: List[str]):
        """Constructor of the class.

        Args:
            researcher_id: ID of the researcher that requests setup
            secagg_id: ID of secagg context element for this setup request
            sequence: unique sequence number of setup request
            parties: List of parties participating to the secagg context element setup
        """
        super().__init__(researcher_id, secagg_id, sequence, parties)
        # add subclass specific init here

    def element(self) -> str:
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
        import time
        time.sleep(4)
        logger.info("Not implemented yet, PUT SECAGG SERVKEY PAYLOAD HERE")
        msg = self._create_secagg_reply('', True)

        return msg


class SecaggBiprimeSetup(SecaggSetup):
    """
    Sets up a biprime Secure Aggregation context element on the node side.

        Args:
            researcher_id: ID of the researcher that requests setup
            secagg_id: ID of secagg context element for this setup request
            sequence: unique sequence number of setup request
            parties: List of parties participating to the secagg context element setup
    """
    def __init__(
            self,
            researcher_id: str,
            secagg_id: str,
            sequence: int,
            parties: List[str]):
        """Constructor of the class.
        """
        super().__init__(researcher_id, secagg_id, sequence, parties)
        # add subclass specific init here

    def element(self) -> str:
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
        import time
        time.sleep(6)
        logger.info("Not implemented yet, PUT SECAGG BIPRIME PAYLOAD HERE")
        msg = self._create_secagg_reply('', True)

        return msg
