"""Secure Aggregation setup on the node"""
from typing import List

from fedbiomed.common.constants import SecaggElementTypes
from fedbiomed.common.message import NodeMessages, SecaggReply
from fedbiomed.common.logger import logger

from fedbiomed.node.environ import environ

class SecaggSetup:
    """
    Sets up a Secure Aggregation context element on the node side.
    """

    def __init__(
            self,
            researcher_id: str,
            secagg_id: str,
            element: SecaggElementTypes,
            parties: List[str]):
        """Constructor of the class.

        Args:
            researcher_id: ID of the researcher that requests setup
            secagg_id: ID of secagg context element for this setup request
            element: Type of secagg context element
            parties: List of parties participating to the secagg context element setup
        """
        # we can suppose input was properly checked before instantiating this object

        self._researcher_id = researcher_id
        self._secagg_id = secagg_id
        self._element = element
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

    def element(self) -> str:
        """Getter for secagg context element type

        Returns:
            secagg context element name
        """
        return self._element.name

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
                'success': success,
                'node_id': environ['NODE_ID'],
                'msg': message,
                'command': 'secagg'
            }
        ).get_dict()

    # TODO: subclass per type
    def setup(self) -> SecaggReply:
        """Set up a secagg context element.

        Returns:
            message to return to the researcher after the setup
        """
        logger.info("PUT SECAGG PAYLOAD HERE")
        msg = self._create_secagg_reply('', True)

        return msg