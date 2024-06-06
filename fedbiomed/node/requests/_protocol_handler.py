# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import asyncio

from fedbiomed.common.message import InnerMessage
from fedbiomed.common.logger import logger
from fedbiomed.common.message import NodeMessages, NodeToNodeMessages

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

    async def handle(self, overlay_msg: dict, inner_msg: InnerMessage) -> None:
        """xxx"""

        if inner_msg.get_param('command') in self._command2method:
            await self._command2method[inner_msg.get_param('command')](overlay_msg, inner_msg)
        else:
            await self._HandlerDefault(overlay_msg, inner_msg)


    # Each handler receives `overlay_msg` and `inner_msg`, returns `None`
    # It may receive an asyncio.CancelledError and handle it to exit cleanly
    #
    #async def _HandlerExample(self, overlay_msg: dict, inner_msg: InnerMessage) -> None:
    #    try:
    #        logger.debug("Normal handler code")
    #    except asyncio.CancelledError:
    #        logger.debug("To be executed if task is cancelled, eg timeout")
    #    finally:
    #        logger.debug("To be executed in any case for finishing cleanly the task")


    async def _HandlerDefault(self, overlay_msg: dict, inner_msg: InnerMessage) -> None:
        """xxx"""
        logger.error(f"Failed processing overlay message, unknown inner command {inner_msg.get_param('command')}. Do nothing.")

    async def _HandlerKeyRequest(self, overlay_msg: dict, inner_msg: InnerMessage) -> None:
        """xxx"""
        try:
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
        except asyncio.CancelledError:
            pass
        else:
            logger.info(f"SENDING OVERLAY message to {inner_msg.get_param('node_id')}: {overlay_resp}")
            self._grpc_controller.send(overlay_resp)


    async def _HandlerKeyReply(self, overlay_msg: dict, inner_msg: InnerMessage) -> None:
        """xxx"""
        try:
            logger.debug("IN HANDLER KEY REPLY")
        except asyncio.CancelledError:
            pass
        finally:
            self._pending_requests.add_reply(inner_msg.get_param('request_id'), inner_msg)


    async def _HandlerDummyInner(self, overlay_msg: dict, inner_msg: InnerMessage) -> None:
        """xxx"""
        try:
            logger.debug("IN HANDLER DUMMY INNER")

            logger.debug(f"GOT A dummy-request {inner_msg}")

            while True:
                logger.debug(f"===== WAIT 1 SECOND IN DUMMY INNER")
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            logger.debug("HANDLER DUMMY REQUEST WAS CANCELLED")
        finally:
            logger.debug("HANDLER DUMMY REQUEST FINAL ACTION START")
            await asyncio.sleep(4)
            logger.debug("HANDLER DUMMY REQUEST FINAL ACTION COMPLETE")
