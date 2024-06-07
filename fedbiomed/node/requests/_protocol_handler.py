# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
import asyncio
import inspect

from fedbiomed.common.message import InnerMessage, OverlaySend, NodeMessages, NodeToNodeMessages
from fedbiomed.common.logger import logger

from fedbiomed.node.environ import environ
from ._overlay import format_outgoing_overlay
from ._pending_requests import PendingRequests

from fedbiomed.transport.controller import GrpcController


class ProtocolHandler:
    """xxx"""

    def __init__(self, grpc_controller: GrpcController, pending_requests: PendingRequests) -> None:
        """xxx"""
        self._grpc_controller = grpc_controller
        self._pending_requests = pending_requests

        self._command2method = {
            'key-request': self._HandlerKeyRequest,
            'key-reply': self._HandlerKeyReply,
            'dummy-inner': self._HandlerDummyInner,
        }

        self._command2final = {
            'key-request': self._FinalKeyRequest,
            'key-reply': self._FinalKeyReply,
            'dummy-inner': self._FinalDummyInner,
        }

    async def handle(self, overlay_msg: dict, inner_msg: InnerMessage) -> Optional[dict]:
        """xxx"""

        if inner_msg.get_param('command') in self._command2method:
            return await self._command2method[inner_msg.get_param('command')](overlay_msg, inner_msg)
        else:
            return await self._HandlerDefault(overlay_msg, inner_msg)

    async def final(self, command, **kwargs) -> None:
        """xxx"""
        if command in self._command2final:
            # Useful ? Allow omitting some arguments, automatically add them with None value
            expected_args = dict(inspect.signature(self._command2final[command]).parameters).keys()
            kwargs.update({arg: None for arg in expected_args if arg not in kwargs})

            await self._command2final[command](**kwargs)

    async def _HandlerDefault(self, overlay_msg: dict, inner_msg: InnerMessage) -> None:
        """xxx"""
        logger.error(
            f"Failed processing overlay message, unknown inner command {inner_msg.get_param('command')}. Do nothing.")

    # Each message type must have a handler.
    # A handler receives `overlay_msg` and `inner_msg`, returns a dict
    # which will be passed as `**kwargs` to the `final()` - types must match !
    # It may receive an asyncio.CancelledError
    #
    # Each message type optionally has a final.
    # It executes only if the `handler()` completed without being cancelled
    # It won't be interrupted by an asyncio.CancelledError
    # If no `final()` exist, no action is taken after cancelling or completing the `handler()`
    #
    # async def _HandlerExample(self, overlay_msg: dict, inner_msg: InnerMessage) -> Any:
    #     logger.debug("Normal handler code that can be cancelled")
    #     return { 'value: 3 }
    # async def _FinalExample(self, value: int) -> None:
    #         logger.debug(f"Final code than cannot be cancelled. Received {value}")

    async def _HandlerKeyRequest(self, overlay_msg: dict, inner_msg: InnerMessage) -> dict:
        """xxx"""
        logger.debug("IN HANDLER KEY REQUEST")

        # TEST: implement arbitrary delay
        import random
        delay = random.randrange(1, 15)
        for i in range(delay):
            logger.debug(f"===== WAIT 1 SECOND IN PROTOCOL MANAGER {i+1}/{delay}")
            await asyncio.sleep(1)

        # For real use: catch FedbiomedNodeToNodeError when calling `format_outgoing_overlay`
        inner_resp = NodeToNodeMessages.format_outgoing_message(
            {
                'request_id': inner_msg.get_param('request_id'),
                'node_id': environ['NODE_ID'],
                'dest_node_id': inner_msg.get_param('node_id'),
                'dummy': f"KEY REPLY INNER from {environ['NODE_ID']}",
                'secagg_id': inner_msg.get_param('secagg_id'),
                'command': 'key-reply'
            })
        overlay_resp = NodeMessages.format_outgoing_message(
            {
                'researcher_id': overlay_msg['researcher_id'],
                'node_id': environ['NODE_ID'],
                'dest_node_id': inner_msg.get_param('node_id'),
                'overlay': format_outgoing_overlay(inner_resp),
                'command': 'overlay-send'
            })

        return { 'inner_msg': inner_msg, 'overlay_resp': overlay_resp }

    async def _FinalKeyRequest(self, inner_msg: InnerMessage, overlay_resp: OverlaySend) -> None:
        """xxx"""
        logger.debug(f"FINAL REQUEST {inner_msg} {overlay_resp}")
        logger.info(f"SENDING OVERLAY message to {inner_msg.get_param('node_id')}: {overlay_resp}")
        self._grpc_controller.send(overlay_resp)


    async def _HandlerKeyReply(self, overlay_msg: dict, inner_msg: InnerMessage) -> dict:
        """xxx"""
        return { 'inner_msg': inner_msg }

    async def _FinalKeyReply(self, inner_msg: InnerMessage) -> None:
        """xxx"""
        logger.debug(f"FINAL REPLY {inner_msg}")
        self._pending_requests.add_reply(inner_msg.get_param('request_id'), inner_msg)


    async def _HandlerDummyInner(self, overlay_msg: dict, inner_msg: InnerMessage) -> None:
        """xxx"""
        logger.debug("IN HANDLER DUMMY INNER")
        logger.debug(f"GOT A dummy-request {inner_msg}")

        logger.debug("HANDLER DUMMY REQUEST  START")
        await asyncio.sleep(3600)
        logger.debug("HANDLER DUMMY REQUEST COMPLETE")

    async def _FinalDummyInner(self):
        """xxx"""
        logger.debug("HANDLER DUMMY REQUEST FINAL ACTION START")
        await asyncio.sleep(4)
        logger.debug("HANDLER DUMMY REQUEST FINAL ACTION COMPLETE")
