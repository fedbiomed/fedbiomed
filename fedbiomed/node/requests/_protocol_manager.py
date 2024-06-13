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
OVERLAY_MESSAGE_PROCESS_TIMEOUT = 60


class _ProtocolAsyncManager:
    """Background async thread for handling node to node messages received by a node."""

    def __init__(self, grpc_controller: GrpcController, pending_requests: PendingRequests) -> None:
        """Class constructor.

        Args:
            grpc_controller: object managing the communication with other components
            pending_requests: object for receiving overlay node to node messages
        """
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
        """Callback launched when a task completes.

        Args:
            task: task object
        """
        # needed to perform async operations from `add_done_callback()`
        asyncio.create_task(self._remove_finished_task_async(task))

    async def _remove_finished_task_async(self, task: asyncio.Task) -> None:
        """Main function for short-lived task launched after a message processing task completes.

        Args:
            task: task object
        """
        async with self._active_tasks_lock:
            task_name = task.get_name()
            if task_name in self._active_tasks:
                del self._active_tasks[task_name]
            else:
                # don't raise exception
                logger.error(f"{ErrorNumbers.FB324}: task already finished {task_name}")

    async def _clean_active_tasks(self) -> None:
        '''Main function for background task cleaning active task list.

        Cancels tasks that reached a timeout and did not yet complete the main handler.
        '''
        while True:
            await asyncio.sleep(1)

            current_time = time.time()
            async with self._active_tasks_lock:
                for task_name in self._active_tasks:
                    if self._active_tasks[task_name]['start_time'] + OVERLAY_MESSAGE_PROCESS_TIMEOUT < current_time \
                            and not self._active_tasks[task_name]['finally']:
                        # A task is expired: cancel it
                        # Entry from the table will be cleaned by the `add_done_callback()`
                        # after completing task `final()`
                        self._active_tasks[task_name]['task'].cancel()

                        # we cannot just rely on `task.cancel()` as it may re-cancel a task in its `final()` clause
                        # if it does not complete it before next loop of this function
                        # Basically, we want to issue *once* a cancel() to a task
                        # and then trust it to properly complete
                        self._active_tasks[task_name]['finally'] = True

    async def _run_async(self) -> None:
        """Main async function for the protocol manager background thread."""

        self._loop = asyncio.get_running_loop()

        # task for cleaning the active tasks
        self._task_clean_active_tasks = asyncio.create_task(self._clean_active_tasks())

        while True:
            msg = await self._queue.get()
            self._queue.task_done()

            async with self._active_tasks_lock:
                # TODO: add test for maximum number of tasks ?

                # timeout for tasks in handled via `_clean_active_tasks` task
                task_msg = asyncio.create_task(self._overlay_message_process(msg))
                self._active_tasks[task_msg.get_name()] = {
                    'start_time': time.time(),
                    'task': task_msg,
                    'finally': False,
                }
                #
                task_msg.add_done_callback(self._remove_finished_task)

    async def _submit(self, msg) -> None:
        """Submits a received message to the protocol manager for processing.

        Args:
            msg: received message
        """
        try:
            await self._queue.put_nowait(msg)
        except asyncio.QueueFull as e:
            logger.error(
                f"{ErrorNumbers.FB324}: Failed submitting message to protocol manager. Discard message. "
                f"Exception: {type(e).__name__}. Error message: {e}")
            # don't raise exception

    async def _overlay_message_process(self, overlay_msg: dict) -> None:
        """Main function for a task processing a received message.

        Args:
            overlay_msg: received outer message
        """

        try:
            try:
                if overlay_msg['dest_node_id'] != environ['NODE_ID']:
                    logger.error(
                        f"{ErrorNumbers.FB324}: Node {environ['NODE_ID']} received an overlay message "
                        f"sent to {overlay_msg['dest_node_id']}. Maybe malicious activity. Ignore message."
                    )
                    return
                inner_msg = format_incoming_overlay(overlay_msg['overlay'])

                finally_kwargs = await self._protocol_handler.handle(overlay_msg, inner_msg)
                # in case nothing is returned from the handler
                if finally_kwargs is None:
                    finally_kwargs = {}

                # Prevent cancelling from now on
                # There won't be a race condition with the `_clean_active_tasks`:
                # if we get the lock, then it cannot `cancel()` this task, as it need to get 
                # the lock for that
                async with self._active_tasks_lock:
                    self._active_tasks[asyncio.current_task().get_name()]['finally'] = True

            except asyncio.CancelledError as e:
                logger.error(
                    f"{ErrorNumbers.FB324}: Task {asyncio.current_task().get_name()} was cancelled before completing. "
                    f"Error message: {e}. Overlay message: {overlay_msg}"
                )
            else:
                await self._protocol_handler.final(inner_msg.get_param('command'), **finally_kwargs)

        except Exception as e:
            logger.error(
                f"{ErrorNumbers.FB324}: Failed processing overlay message. Exception: {type(e).__name__}. "
                f"Error message: {e}. Overlay message: {overlay_msg}")
            # don't raise exception


class ProtocolManager(_ProtocolAsyncManager):
    """Handles node to node messages received by a node."""

    def __init__(self, grpc_controller: GrpcController, pending_requests: PendingRequests) -> None:
        """Class constructor.

        Args:
            grpc_controller: object managing the communication with other components
            pending_requests: object for receiving overlay node to node messages
        """
        super().__init__(grpc_controller, pending_requests)

        self._thread = Thread(target=self._run, args=(), daemon=True)


    def _run(self) -> None:
        """Main function for the protocol manager background thread."""
        try:
            asyncio.run(self._run_async())
        except Exception as e:
            logger.critical(
                f"Failed launching node protocol manager. Exception: {type(e).__name__}. Error message: {e}")
            raise e


    def start(self) -> None:
        """Starts the protocol manager."""
        self._thread.start()


    def submit(self, msg: dict) -> None:
        """Submits a received message to the protocol manager for processing.

        Args:
            msg: received message

        Raises:
            FedbiomedNodeToNodeError: bad message type or value.
        """
        # Protocol manager currently handles only node to node messages
        # Conceived to be later extended for other messages processing, during node redesign
        if not isinstance(msg, dict) or 'command' not in msg or msg['command'] != 'overlay-forward':
            raise FedbiomedNodeToNodeError(
                f'{ErrorNumbers.FB324.value}: protocol manager needs a node to node message')

        try:
            asyncio.run_coroutine_threadsafe(self._submit(msg), self._loop)
        except Exception as e:
            logger.critical(
                f"Failed submitting message to protocol manager. Exception: {type(e).__name__}. Error message: {e}")
            raise e