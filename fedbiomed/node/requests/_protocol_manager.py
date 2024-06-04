# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from threading import Thread
import asyncio
import time

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedNodeToNodeError
from fedbiomed.common.logger import logger
from fedbiomed.common.message import NodeMessages, NodeToNodeMessages

from fedbiomed.node.environ import environ
from ._overlay import format_outgoing_overlay, format_incoming_overlay
from ._pending_requests import PendingRequests

from fedbiomed.transport.controller import GrpcController


MAX_PROTOCOL_MANAGER_QUEUE_SIZE = 1000


class _ProtocolAsyncManager:
    """xxx"""

    def __init__(self, grpc_controller: GrpcController, pending_requests: PendingRequests) -> None:
        """xxx"""
        self._grpc_controller = grpc_controller
        self._pending_requests = pending_requests

        self._queue = asyncio.Queue(MAX_PROTOCOL_MANAGER_QUEUE_SIZE)
        self._loop = None
        self._active_tasks = {}

        # exclusive access to `self._active_tasks`
        self._active_tasks_lock = asyncio.Lock()

    def _clean_finished_task(self, task: asyncio.Task) -> None:
        """xxx"""
        asyncio.create_task(self._change_active_after_task(task))

    async def _change_active_after_task(self, task: asyncio.Task) -> None:
        """xxx"""
        async with self._active_tasks_lock:
            logger.debug(f"===== BEFORE {self._active_tasks}")
            task_name = task.get_name()
            if task.get_name() in self._active_tasks:
                del self._active_tasks[task_name]
            else:
                # don't raise exception
                logger.error(f"{ErrorNumbers.FB324}: task already finished {task_name}")
            logger.debug(f"===== AFTER {self._active_tasks}")

    async def _start(self) -> None:
        """xxx"""
        self._loop = asyncio.get_running_loop()

        while True:
            msg = await self._queue.get()
            self._queue.task_done()

            # TODO : add timeout to tasks. Cancel task + issue warning message
            # that message could not be fully processed before timeout

            async with self._active_tasks_lock:
                logger.debug(f"===== ACTIVE TASKS {list(self._active_tasks.keys())}")

                # TODO: test for maximum number of tasks ?

                task_msg = asyncio.create_task(self._overlay_message_process(msg))
                # TODO `time` to be used for timeouts
                self._active_tasks[task_msg.get_name()] = {
                    'start_time': time.time(),
                    'task': task_msg,
                }
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
            logger.info(f"===== RECEIVED OVERLAY MESSAGE {overlay_msg}")
            logger.info(f"===== RECEIVED INNER MESSAGE {inner_msg}")

            #
            # TODO: remove, temporary test - replace with parent/child class per message type
            #
            if inner_msg.get_param('command') == 'key-request':

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
                logger.info(f"SENDING OVERLAY message to {inner_msg.get_param('node_id')}: {overlay_resp}")
                self._grpc_controller.send(overlay_resp)

            if inner_msg.get_param('command') == 'key-reply':
                self._pending_requests.add_reply(inner_msg.get_param('request_id'), inner_msg)

            if inner_msg.get_param('command') == 'dummy-inner':
                logger.debug(f"GOT A dummy-request {inner_msg}")

        except Exception as e:
            logger.critical(
                f"Failed processing overlay message. Exception: {type(e).__name__}. Error message: {e}. "
                f"Overlay message: {overlay_msg}")
            # don't raise exception


class ProtocolManager(_ProtocolAsyncManager):
    """xxx"""

    def __init__(self, grpc_controller: GrpcController, pending_requests: PendingRequests) -> None:
        """xxx"""
        super().__init__(grpc_controller, pending_requests)

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


    def submit(self, msg: dict) -> None:
        """xxx"""
        # Protocol manager currently handles only node2node messages
        # TODO: extend for other messages during node redesign
        if not isinstance(msg, dict) or 'command' not in msg or msg['command'] != 'overlay-forward':
            raise FedbiomedNodeToNodeError(
                f'{ErrorNumbers.FB324.value}: protocol manager needs a node to node message')

        try:
            asyncio.run_coroutine_threadsafe(self._submit(msg), self._loop)
        except Exception as e:
            logger.critical(
                f"Failed submitting message to protocol manager. Exception: {type(e).__name__}. Error message: {e}")
            raise e
