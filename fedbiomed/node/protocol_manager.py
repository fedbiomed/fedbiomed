# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from threading import Thread
import asyncio
import time

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.logger import logger
from fedbiomed.common.message import NodeMessages, NodeToNodeMessages

from fedbiomed.node.environ import environ
from fedbiomed.node.overlay import format_outgoing_overlay, format_incoming_overlay

from fedbiomed.transport.controller import GrpcController


MAX_PROTOCOL_MANAGER_QUEUE_SIZE = 1000


class _ProtocolAsyncManager:
    """xxx"""

    def __init__(self, grpc_controller: GrpcController) -> None:
        """xxx"""
        self._grpc_controller = grpc_controller
        self._active_tasks = {}
        self._queue = asyncio.Queue(MAX_PROTOCOL_MANAGER_QUEUE_SIZE)
        self._loop = None

    def _clean_finished_task(self, task: asyncio.Task) -> None:
        """xxx"""
        del self._active_tasks[task.get_name()]

    async def _start(self) -> None:
        """xxx"""
        self._loop = asyncio.get_running_loop()

        while True:
            msg = await self._queue.get()
            self._queue.task_done()

            logger.debug(f"******* ACTIVE TASKS {list(self._active_tasks.keys())}")

            task_msg = asyncio.create_task(self._overlay_message_process(msg))
            # TODO `time` to be used for timeouts
            self._active_tasks[task_msg.get_name()] = time.time()
            #
            task_msg.add_done_callback(self._clean_finished_task)

    async def _submit(self, msg) -> None:
        try:
            await self._queue.put_nowait(msg)
        except asyncio.QueueFull as e:
            logger.critical(
                "Failed submitting message to protocol manager. Discard message. "
                f"Exception: {type(e).__name__}. Error message: {e}")
            # don't raise exception

    async def _overlay_message_process(self, overlay_msg: dict) -> None:
        """xxx"""

        try:
            if overlay_msg['dest_node_id'] != environ['NODE_ID']:
                logger.error(
                    f"{ErrorNumbers.FB324}: node {environ['NODE_ID']} received an overlay message "
                    f"sent to {overlay_msg['dest_node_id']}. Maybe malicious activity. Ignore message."
                )
                return
            inner_msg = format_incoming_overlay(overlay_msg['overlay'])

            # TODO: implement payload for overlay in sub-function
            #
            logger.info(f"RECEIVED OVERLAY MESSAGE {overlay_msg}")
            logger.info(f"RECEIVED INNER MESSAGE {inner_msg}")

            #
            # TODO: remove, temporary test
            #
            if inner_msg.get_param('command') == 'key-request':

                # TEST: implement arbitrary delay
                import random
                delay = random.randrange(1, 20)
                for i in range(delay):
                    logger.debug(f"*** WAIT 1 SECOND {i+1}/{delay}")
                    await asyncio.sleep(1)

                # For real use: catch FedbiomedNodeToNodeError when calling `format_outgoing_overlay`
                inner_resp = NodeToNodeMessages.format_outgoing_message(
                    {
                        'request_id': inner_msg.get_param('request_id'),
                        'node_id': environ['NODE_ID'],
                        'dest_node_id': inner_msg.get_param('node_id'),
                        'dummy': f"DUMMY INNER KEY REPLY from {environ['NODE_ID']}",
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
                logger.info(f"SENDING OVERLAY message to {inner_msg.get_param('node_id')}: {overlay_resp}")
                self._grpc_controller.send(overlay_resp)

        except Exception as e:
            logger.critical(
                f"Failed processing overlay message. Exception: {type(e).__name__}. Error message: {e}. "
                f"Overlay message: {overlay_msg}")
            # don't raise exception


class ProtocolManager(_ProtocolAsyncManager):
    """xxx"""

    def __init__(self, grpc_controller: GrpcController) -> None:
        """xxx"""
        super().__init__(grpc_controller)

        self._thread = Thread(target=self._run, args=(), daemon=True)


    def _run(self) -> None:
        """xxx"""
        try:
            asyncio.run(self._start())
        except Exception as e:
            logger.critical(
                f"Failed launching node protocol manager. Exception: {type(e).__name__}. Error message: {e}")
            raise e


    def start(self) -> None:
        """xxx"""
        self._thread.start()


    def submit(self, msg) -> None:
        """xxx"""
        try:
            asyncio.run_coroutine_threadsafe(self._submit(msg), self._loop)
        except Exception as e:
            logger.critical(
                f"Failed submitting message to protocol manager. Exception: {type(e).__name__}. Error message: {e}")
            raise e

