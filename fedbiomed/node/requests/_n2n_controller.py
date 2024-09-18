# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
import asyncio
import inspect

from fedbiomed.common.constants import ErrorNumbers, TIMEOUT_NODE_TO_NODE_REQUEST
from fedbiomed.common.message import (
    Message,
    KeyRequest,
    KeyReply,
    InnerMessage,
    OverlayMessage
)
from fedbiomed.common.logger import logger
from fedbiomed.common.synchro import EventWaitExchange

from fedbiomed.node.environ import environ
from ._overlay import format_outgoing_overlay

from fedbiomed.transport.controller import GrpcController


class NodeToNodeController:
    """Defines the controller for protocol messages processed by the node to node router



    Each message type must have a handler.
    A handler receives `overlay_msg` and `inner_msg`, returns a dict
    which will be passed as `**kwargs` to the `final()` - types must match !
    It may receive an asyncio.CancelledError

    Each message type optionally has a final.
    It executes only if the `handler()` completed without being cancelled
    It won't be interrupted by an asyncio.CancelledError
    If no `final()` exist, no action is taken after cancelling or completing the `handler()`

    async def _HandlerExample(self, overlay_msg: dict, inner_msg: InnerMessage) -> Any:
        logger.debug("Normal handler code that can be cancelled")
        return { 'value: 3 }
    async def _FinalExample(self, value: int) -> None:
            logger.debug(f"Final code than cannot be cancelled. Received {value}")

    """

    def __init__(
            self,
            grpc_controller: GrpcController,
            pending_requests: EventWaitExchange,
            controller_data: EventWaitExchange,
    ) -> None:
        """Constructor of the class.

        Args:
            grpc_controller: object managing the communication with other components
            pending_requests: object for receiving overlay node to node messages
            controller_data: object for sharing data
        """
        self._grpc_controller = grpc_controller
        self._pending_requests = pending_requests
        self._controller_data = controller_data

        self._command2method = {
            KeyRequest.__name__: self._HandlerKeyRequest,
            KeyReply.__name__: self._HandlerKeyReply,
        }

        self._command2final = {
            KeyRequest.__name__: self._FinalKeyRequest,
            KeyReply.__name__: self._FinalKeyReply,
        }

    async def handle(self, overlay_msg: Message, inner_msg: InnerMessage) -> Optional[dict]:
        """Calls the handler for processing a received message protocol.

        If it does not exist, call the default handler to trigger an error.

        Main part of the processing which can be interrupted if the processing takes too long.

        Args:
            overlay_msg: Outer message for node to node communication
            inner_msg: Unpacked inner message from the outer message

        Returns:
            A dict of the `kwargs` expected by the corresponding `final()` handler for this message.
                Empty dict or `None` if no `kwargs` expected of no final handler
        """

        if inner_msg.__name__ in self._command2method:
            return await self._command2method[inner_msg.__name__](overlay_msg, inner_msg)

        return await self._HandlerDefault(overlay_msg, inner_msg)

    async def final(self, message, **kwargs) -> None:
        """Calls the final processing for a received message protocol.

        This handler is optional, it may not be declared for a message.

        Should be called only if the handler completed without being interrupted.
        Cannot be interrupted, thus should not launch treatment that may hang.

        Args:
            kwargs: Specific arguments for this message final handler
        """
        if message in self._command2final:
            # Useful ? Allow omitting some arguments, automatically add them with None value
            expected_args = dict(inspect.signature(self._command2final[message]).parameters).keys()
            kwargs.update({arg: None for arg in expected_args if arg not in kwargs})

            await self._command2final[message](**kwargs)

    async def _HandlerDefault(self, overlay_msg: dict, inner_msg: InnerMessage) -> None:
        """Handler called if the handler for this message is missing.

        Args:
            overlay_msg: Outer message for node to node communication
            inner_msg: Unpacked inner message from the outer message

        Returns:
            None
        """

        logger.error(
            f"{ErrorNumbers.FB324}: Failed processing overlay message, unknown inner command "
            f"{inner_msg.__class__.__name__}. Do nothing.")

    async def _HandlerKeyRequest(self, overlay_msg: dict, inner_msg: InnerMessage) -> dict:
        """Handler called for KeyRequest message.

        Args:
            overlay_msg: Outer message for node to node communication
            inner_msg: Unpacked inner message from the outer message

        Returns:
            A `dict` with overlay reply message
        """
        # Wait until node has generated its DH keypair
        all_received, data = self._controller_data.wait(
            [inner_msg.get_param('secagg_id')],
            TIMEOUT_NODE_TO_NODE_REQUEST
        )

        # Don't send reply message if the public key is not available after a timeout
        if not all_received:
            return None

        # we assume the data is properly formatted
        inner_resp = KeyReply(
            request_id=inner_msg.request_id,
            node_id=environ['NODE_ID'],
            dest_node_id=inner_msg.node_id,
            public_key=data[0]['public_key'],
            secagg_id=inner_msg.secagg_id)

        overlay_resp = OverlayMessage(
            researcher_id=overlay_msg.researcher_id,
            node_id=environ['NODE_ID'],
            dest_node_id=inner_msg.node_id,
            overlay=format_outgoing_overlay(inner_resp))


        return { 'overlay_resp': overlay_resp }




    async def _FinalKeyRequest(self, overlay_resp: Optional[OverlayMessage]) -> None:
        """Final handler called for KeyRequest message.

        Args:
            overlay_resp: overlay reply message to send
        """
        if isinstance(overlay_resp, OverlayMessage):
            self._grpc_controller.send(overlay_resp)

    async def _AdditiveSSharingRequest(self, overlay_resp: Optional[OverlayMessage]):
        """Final handler called for AdditiveSSharingRequest message.

        Args:
            overlay_resp: overlay reply message to send
        """

        from_ = request.node_id
        # Wait until node has generated its share for given secagg id
        all_received, data = self._controller_data.wait(
            [request.secagg_id],
            TIMEOUT_NODE_TO_NODE_REQUEST
        )

        if not all_received:
            return None

        share = data[0]["shares"].get(from_)

        return data


    async def _HandlerKeyReply(self, overlay_msg: dict, inner_msg: InnerMessage) -> dict:
        """Handler called for KeyReply message.

        Args:
            overlay_msg: Outer message for node to node communication
            inner_msg: Unpacked inner message from the outer message

        Returns:
            A `dict` with received inner message
        """
        return { 'inner_msg': inner_msg }

    async def _FinalKeyReply(self, inner_msg: InnerMessage) -> None:
        """Final handler called for KeyReply message.

        Args:
            inner_msg: received inner message
        """

        self._pending_requests.event(inner_msg.get_param('request_id'), inner_msg)


