# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import inspect
from typing import Optional

from fedbiomed.common.constants import ErrorNumbers, TIMEOUT_NODE_TO_NODE_REQUEST
from fedbiomed.common.logger import logger
from fedbiomed.common.message import (
    AdditiveSSharingReply,
    AdditiveSSharingRequest,
    ChannelSetupRequest,
    ChannelSetupReply,
    InnerMessage,
    KeyReply,
    KeyRequest,
    OverlayMessage,
)
from fedbiomed.common.synchro import EventWaitExchange

from fedbiomed.transport.controller import GrpcController

from ._overlay import OverlayChannel


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
            node_id: str,
            grpc_controller: GrpcController,
            overlay_channel: OverlayChannel,
            pending_requests: EventWaitExchange,
            controller_data: EventWaitExchange,
    ) -> None:
        """Constructor of the class.

        Args:
            node_id: ID of the node.
            grpc_controller: object managing the communication with other components
            overlay_channel: layer for managing overlay message send and receive
            pending_requests: object for receiving overlay node to node messages
            controller_data: object for sharing data
        """

        self._node_id = node_id
        self._grpc_controller = grpc_controller
        self._overlay_channel = overlay_channel
        self._pending_requests = pending_requests
        self._controller_data = controller_data

        self._command2method = {
            KeyRequest.__name__: self._HandlerKeyRequest,
            KeyReply.__name__: self._HandlerKeyReply,
            AdditiveSSharingRequest.__name__: self._AdditiveSSharingRequest,
            AdditiveSSharingReply.__name__: self._HandlerAdditiveSSharingReply,
            ChannelSetupRequest.__name__: self._HandlerChannelRequest,
            ChannelSetupReply.__name__: self._HandlerKeyReply,
        }

        self._command2final = {
            KeyRequest.__name__: self._FinalKeyRequest,
            KeyReply.__name__: self._FinalKeyReply,
            AdditiveSSharingRequest.__name__: self._FinalAdditiveSSharingRequest,
            AdditiveSSharingReply.__name__: self._FinalAdditiveSSharingReply,
            ChannelSetupRequest.__name__: self._FinalKeyRequest,
            ChannelSetupReply.__name__: self._FinalChannelReply,
        }

    async def handle(
        self, overlay_msg: OverlayMessage, inner_msg: InnerMessage
    ) -> Optional[dict]:
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
            return await self._command2method[inner_msg.__name__](
                overlay_msg, inner_msg
            )

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
            expected_args = dict(
                inspect.signature(self._command2final[message]).parameters
            ).keys()
            kwargs.update({arg: None for arg in expected_args if arg not in kwargs})

            await self._command2final[message](**kwargs)

    async def _HandlerDefault(  # pylint: disable=C0103
        self, overlay_msg: OverlayMessage, inner_msg: InnerMessage
    ) -> None:
        """Handler called if the handler for this message is missing.

        Args:
            overlay_msg: Outer message for node to node communication
            inner_msg: Unpacked inner message from the outer message

        Returns:
            None
        """

        logger.error(
            f"{ErrorNumbers.FB324}: Failed processing overlay message, unknown inner command "
            f"{inner_msg.__class__.__name__}. Do nothing."
        )

    async def _HandlerChannelRequest(self, overlay_msg: OverlayMessage, inner_msg: InnerMessage) -> dict:
        """Handler called for ChannelSetupRequest message.

        Args:
            overlay_msg: Outer message for node to node communication
            inner_msg: Unpacked inner message from the outer message

        Returns:
            A `dict` with overlay reply message
        """
        # we assume the data is properly formatted
        inner_resp = ChannelSetupReply(
            request_id=inner_msg.request_id,
            node_id=self._node_id,
            dest_node_id=inner_msg.node_id,
            public_key=await self._overlay_channel.get_local_public_key(inner_msg.node_id))

        overlay, salt, nonce = await self._overlay_channel.format_outgoing_overlay(
            inner_resp,
            overlay_msg.researcher_id,
            True
        )
        overlay_resp = OverlayMessage(
            researcher_id=overlay_msg.researcher_id,
            node_id=self._node_id,
            dest_node_id=inner_msg.node_id,
            overlay=overlay,
            setup=True,
            salt=salt,
            nonce=nonce)

        return { 'overlay_resp': overlay_resp }

    async def _FinalChannelReply(self, inner_msg: InnerMessage) -> None:
        """Final handler called for ChannelSetupReply message.

        Args:
            inner_msg: received inner message
        """
        if not await self._overlay_channel.set_distant_key(
            inner_msg.node_id,
            inner_msg.public_key,
            inner_msg.request_id,
        ):
            logger.warning(
                f"{ErrorNumbers.FB324}: Received channel key of unregistered "
                f"peer node {inner_msg.node_id} "
                f"or reply to non existing request {inner_msg.request_id}. "
                f"Distant node may be confused or it may be an attack."
            )

    async def _HandlerKeyRequest(  # pylint: disable=C0103
        self, overlay_msg: OverlayMessage, inner_msg: InnerMessage
    ) -> dict:
        """Handler called for KeyRequest message.

        Args:
            overlay_msg: Outer message for node to node communication
            inner_msg: Unpacked inner message from the outer message

        Returns:
            A `dict` with overlay reply message
        """
        # Wait until node has generated its DH keypair
        all_received, data = self._controller_data.wait(
            [inner_msg.secagg_id],
            TIMEOUT_NODE_TO_NODE_REQUEST
        )

        # Don't send reply message if the public key is not available after a timeout
        if not all_received:
            return None

        # we assume the data is properly formatted
        inner_resp = KeyReply(
            request_id=inner_msg.request_id,
            node_id=self._node_id,
            dest_node_id=inner_msg.node_id,
            public_key=data[0]["public_key"],
            secagg_id=inner_msg.secagg_id,
        )

        overlay, salt, nonce = await self._overlay_channel.format_outgoing_overlay(inner_resp, overlay_msg.researcher_id)
        overlay_resp = OverlayMessage(
            researcher_id=overlay_msg.researcher_id,
            node_id=self._node_id,
            dest_node_id=inner_msg.node_id,
            overlay=overlay,
            setup=False,
            salt=salt,
            nonce=nonce)

        return { 'overlay_resp': overlay_resp }

    async def _FinalKeyRequest(  # pylint: disable=C0103
        self, overlay_resp: Optional[OverlayMessage]
    ) -> None:
        """Final handler called for KeyRequest message.

        Args:
            overlay_resp: overlay reply message to send
        """
        if isinstance(overlay_resp, OverlayMessage):
            self._grpc_controller.send(overlay_resp)

    async def _AdditiveSSharingRequest(  # pylint: disable=C0103
        self, overlay_msg: OverlayMessage, request: Optional[InnerMessage]
    ):
        """Final handler called for AdditiveSSharingRequest message.

        Args:
            overlay_resp: overlay reply message to send
        """

        from_ = request.node_id
        # Wait until node has generated its share for given secagg id
        all_received, data = self._controller_data.wait(
            [request.secagg_id], TIMEOUT_NODE_TO_NODE_REQUEST
        )

        if not all_received:
            return None

        share = data[0]["shares"].get(from_)

        inner_resp = AdditiveSSharingReply(
            request_id=request.request_id,
            node_id=self._node_id,
            dest_node_id=from_,
            secagg_id=request.secagg_id,
            share=share,
        )

        overlay, salt, nonce = await self._overlay_channel.format_outgoing_overlay(inner_resp, overlay_msg.researcher_id)
        overlay_resp = OverlayMessage(
            researcher_id=overlay_msg.researcher_id,
            node_id=self._node_id,
            dest_node_id=request.node_id,
            overlay=overlay,
            setup=False,
            salt=salt,
            nonce=nonce,
        )

        return {"overlay_resp": overlay_resp}

    async def _FinalAdditiveSSharingRequest(  # pylint: disable=C0103
        self, overlay_resp: Optional[OverlayMessage]
    ) -> None:
        """Final handler called for KeyRequest message.

        Args:
            overlay_resp: overlay reply message to send
        """
        if isinstance(overlay_resp, OverlayMessage):
            self._grpc_controller.send(overlay_resp)

    async def _HandlerKeyReply(  # pylint: disable=C0103
        self, overlay_msg: OverlayMessage, inner_msg: InnerMessage
    ) -> dict:
        """Handler called for KeyReply message.

        Args:
            overlay_msg: Outer message for node to node communication
            inner_msg: Unpacked inner message from the outer message

        Returns:
            A `dict` with received inner message
        """
        return {"inner_msg": inner_msg}

    async def _FinalKeyReply(  # pylint: disable=C0103
        self, inner_msg: InnerMessage
    ) -> None:
        """Final handler called for KeyReply message.

        Args:
            inner_msg: received inner message
        """

        self._pending_requests.event(inner_msg.get_param("request_id"), inner_msg)

    async def _HandlerAdditiveSSharingReply(  # pylint: disable=C0103
        self, overlay_msg: dict, inner_msg: InnerMessage
    ) -> dict:
        """Handler called for AdditiveSharingReply message.

        Args:
            overlay_msg: Outer message for node to node communication
            inner_msg: Unpacked inner message from the outer message

        Returns:
            A `dict` with received inner message
        """
        return {"inner_msg": inner_msg}

    async def _FinalAdditiveSSharingReply(  # pylint: disable=C0103
        self, inner_msg: InnerMessage
    ) -> None:
        """Final handler called for AdditiveSharingReply message.

        Args:
            inner_msg: received inner message
        """
        self._pending_requests.event(inner_msg.request_id, inner_msg)
