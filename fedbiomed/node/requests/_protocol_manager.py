# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from threading import Thread
import asyncio
import time

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedNodeToNodeError
from fedbiomed.common.logger import logger

from fedbiomed.node.environ import environ
from ._overlay import format_incoming_overlay
from ._pending_requests import PendingRequests
from ._protocol_handler import ProtocolHandler

from fedbiomed.transport.controller import GrpcController


# Maximum number of pending messages in the protocol manager input queue
MAX_PROTOCOL_MANAGER_QUEUE_SIZE = 1000

# Maximum duration in seconds of overlay message processing task
OVERLAY_MESSAGE_PROCESS_TIMEOUT = 5


class _ProtocolAsyncManager:
    """xxx"""

    def __init__(self, grpc_controller: GrpcController, pending_requests: PendingRequests) -> None:
        """xxx"""
        self._grpc_controller = grpc_controller
        self._pending_requests = pending_requests

        self._protocol_handler = ProtocolHandler(self._grpc_controller, self._pending_requests)

        self._queue = asyncio.Queue(MAX_PROTOCOL_MANAGER_QUEUE_SIZE)
        self._loop = None

        self._active_tasks = {}
        # exclusive access to `self._active_tasks`
        self._active_tasks_lock = asyncio.Lock()

        self._task_clean_active_tasks = None

    def _remove_finished_task(self, task: asyncio.Task) -> None:
        """xxx"""
        # needed to perform async operations from `add_done_callback()`
        asyncio.create_task(self._remove_finished_task_async(task))

    async def _remove_finished_task_async(self, task: asyncio.Task) -> None:
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

    async def _clean_active_tasks(self) -> None:
        '''xxx TASK METHOD'''

        logger.debug("===== STARTING CLEANING TASK")
        while True:
            await asyncio.sleep(1)

            logger.debug(f"===== CLEANING TASK {self._active_tasks}")

            current_time = time.time()
            async with self._active_tasks_lock:
                for task_name in self._active_tasks:
                    if self._active_tasks[task_name]['start_time'] + OVERLAY_MESSAGE_PROCESS_TIMEOUT < current_time and \
                            not self._active_tasks[task_name]['cancelled']:
                        # A task is expired: cancel it
                        # Entry from the table will be cleaned by the `add_done_callback()` if properly handling the cancel
                        self._active_tasks[task_name]['task'].cancel()

                        # we cannot rely on `task.cancel()` as it may re-cancel a task in its `finally` clause
                        # if it does not complete it before next loop of this function
                        # Basically, we want to issue *once* a cancel() to a task
                        # and then trust it to properly complete
                        self._active_tasks[task_name]['cancelled'] = True

                        logger.debug(f"===== CANCELLING TASK {task_name}")

    async def _start(self) -> None:
        """xxx"""
        self._loop = asyncio.get_running_loop()

        # task for cleaning the active tasks
        self._task_clean_active_tasks = asyncio.create_task(self._clean_active_tasks())

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
                    'cancelled': False,
                }
                #
                task_msg.add_done_callback(self._remove_finished_task)

    async def _submit(self, msg) -> None:
        try:
            await self._queue.put_nowait(msg)
        except asyncio.QueueFull as e:
            logger.error(
                "Failed submitting message to protocol manager. Discard message. "
                f"Exception: {type(e).__name__}. Error message: {e}")
            # don't raise exception

    async def _overlay_message_process(self, overlay_msg: dict) -> None:
        """xxx TASK METHOD"""

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

            await self._protocol_handler.handle(overlay_msg, inner_msg)

        except asyncio.CancelledError as e:
            logger.error(
                f"Task {asyncio.current_task().get_name()} was cancelled before completing. Error message: {e}. "
                f"Overlay message: {overlay_msg}"
            )

        except Exception as e:
            logger.error(
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
