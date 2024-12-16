# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import asyncio
from threading import Thread
import time
from typing import List, Tuple

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.logger import logger
from fedbiomed.common.message import Message, OverlayMessage, InnerMessage
from fedbiomed.common.synchro import EventWaitExchange

from fedbiomed.transport.controller import GrpcController

from ._overlay import OverlayChannel
from ._n2n_controller import NodeToNodeController


# Maximum number of pending messages in the node to node router input queue
MAX_N2N_ROUTER_QUEUE_SIZE = 1000

# Maximum duration in seconds of overlay message processing task
OVERLAY_MESSAGE_PROCESS_TIMEOUT = 60


class _NodeToNodeAsyncRouter:
    """Background async thread for handling node to node messages received by a node."""

    def __init__(
        self,
        node_id: str,
        db: str,
        grpc_controller: GrpcController,
        pending_requests: EventWaitExchange,
        controller_data: EventWaitExchange,
    ) -> None:
        """Class constructor.

        Args:
            node_id: ID of the active node.
            db: Path to database file.
            grpc_controller: object managing the communication with other components
            pending_requests: object for receiving overlay node to node messages
            controller_data: object for sharing data
        """

        self._node_id = node_id
        self._grpc_controller = grpc_controller
        # When implementing multiple researchers, there will probably be
        # one OverlayChannel per researcher
        self._overlay_channel = OverlayChannel(self._node_id, db, self._grpc_controller)
        self._node_to_node_controller = NodeToNodeController(
            self._node_id,
            self._grpc_controller,
            self._overlay_channel,
            pending_requests,
            controller_data
        )

        self._queue = asyncio.Queue(MAX_N2N_ROUTER_QUEUE_SIZE)
        self._loop = None

        self._active_tasks = {}
        # exclusive access to `self._active_tasks`
        self._active_tasks_lock = asyncio.Lock()

        self._task_clean_active_tasks = None

    @property
    def node_id(self):
        """Returns the node id"""
        return  self._node_id

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
        """Main function for background task cleaning active task list.

        Cancels tasks that reached a timeout and did not yet complete the main handler.
        """
        while True:
            await asyncio.sleep(1)

            current_time = time.time()
            async with self._active_tasks_lock:
                for _, task in self._active_tasks.items():
                    if (
                        task["start_time"] + OVERLAY_MESSAGE_PROCESS_TIMEOUT
                        < current_time
                        and not task["finally"]
                    ):
                        # Cancel the task after timeout
                        task["task"].cancel()

                        # Issue *once* a cancel() to a task and then trust it to properly complete
                        task["finally"] = True

    async def _run_async(self) -> None:
        """Main async function for the node to node router background thread."""

        self._loop = asyncio.get_running_loop()

        # task for cleaning the active tasks
        self._task_clean_active_tasks = asyncio.create_task(self._clean_active_tasks())

        while True:
            msg: OverlayMessage = await self._queue.get()
            self._queue.task_done()

            async with self._active_tasks_lock:
                # Current implementation does not test for a maximum number of tasks,
                # only for timeout. Timeout for tasks in handled via `_clean_active_tasks` task
                task_msg = asyncio.create_task(self._overlay_message_process(msg))
                self._active_tasks[task_msg.get_name()] = {
                    "start_time": time.time(),
                    "task": task_msg,
                    "finally": False,
                }
                task_msg.add_done_callback(self._remove_finished_task)

    async def _submit(self, msg: OverlayMessage) -> None:
        """Submits a received message to the node to node router for processing.

        Args:
            msg: received message
        """

        try:
            self._queue.put_nowait(msg)
        # FIXME: Never raises QueueFull maxsize=0 by default
        except asyncio.QueueFull as e:
            logger.error(
                f"{ErrorNumbers.FB324}: Failed submitting message to node to node router. "
                f"Discard message. Exception: {type(e).__name__}. Error message: {e}"
            )

    async def _overlay_message_process(self, overlay_msg: OverlayMessage) -> None:
        """Main function for a task processing a received message.

        Args:
            overlay_msg: received outer message
        """

        try:
            try:
                if overlay_msg.dest_node_id != self._node_id:
                    logger.error(
                        f"{ErrorNumbers.FB324}: Node {self._node_id} received an overlay "
                        f"message sent to {overlay_msg.dest_node_id}. Maybe malicious activity. "
                        "Ignore message."
                    )
                    return
                inner_msg: InnerMessage = await self._overlay_channel.format_incoming_overlay(overlay_msg)

                finally_kwargs = await self._node_to_node_controller.handle(
                    overlay_msg, inner_msg
                )
                # in case nothing is returned from the handler
                if finally_kwargs is None:
                    finally_kwargs = {}

                # Prevent cancelling from now on
                # There won't be a race condition with the `_clean_active_tasks`:
                # if we get the lock, then it cannot `cancel()` this task, as it need to get
                # the lock for that
                async with self._active_tasks_lock:
                    self._active_tasks[asyncio.current_task().get_name()][
                        "finally"
                    ] = True

            except asyncio.CancelledError as e:
                logger.error(
                    f"{ErrorNumbers.FB324}: Task {asyncio.current_task().get_name()} was cancelled "
                    f"before completing. Error message: {e}. Overlay message: "
                    f"{overlay_msg.overlay.__name__}"
                )
            else:
                await self._node_to_node_controller.final(
                    inner_msg.__name__, **finally_kwargs
                )

        except Exception as e:
            logger.error(
                f"{ErrorNumbers.FB324}: Failed processing overlay message. Exception: "
                f"{type(e).__name__}. Error message: {e}. Overlay message: "
                f"{overlay_msg.overlay}"
            )


class NodeToNodeRouter(_NodeToNodeAsyncRouter):
    """Handles node to node messages received by a node."""

    def __init__(
        self,
        node_id: str,
        db: str,
        grpc_controller: GrpcController,
        pending_requests: EventWaitExchange,
        controller_data: EventWaitExchange,
    ) -> None:
        """Class constructor.

        Args:
            node_id: ID of the active node.
            db: Path to database file.
            grpc_controller: object managing the communication with other components
            pending_requests: object for receiving overlay node to node messages
            controller_data: object for sharing data with the controller
        """
        super().__init__(node_id, db, grpc_controller, pending_requests, controller_data)

        self._thread = Thread(target=self._run, args=(), daemon=True)

    def _run(self) -> None:
        """Main function for the node to node router background thread."""
        try:
            asyncio.run(self._run_async())
        except Exception as e:
            logger.critical(
                f"Failed launching node node to node router. Exception: "
                f"{type(e).__name__}. Error message: {e}"
            )
            raise e

    def start(self) -> None:
        """Starts the node to node router."""
        self._thread.start()

    def submit(self, msg: OverlayMessage) -> None:
        """Submits a received message to the node to node router for processing.

        Args:
            msg: received message
        """

        try:
            asyncio.run_coroutine_threadsafe(self._submit(msg), self._loop)
        except Exception as e:
            logger.critical(
                "Failed submitting message to node to node router. "
                f"Exception: {type(e).__name__}. Error message: {e}"
            )
            raise e


    def format_outgoing_overlay(self, message: InnerMessage, researcher_id: str) -> \
            Tuple[bytes, bytes, bytes]:
        """Creates an overlay message payload from an inner message.

        Serialize, crypt, sign the inner message

        Args:
            message: Inner message to send as overlay payload
            researcher_id: unique ID of researcher connecting the nodes
            setup: False for sending a message over the channel, True for a message
                setting up the channel

        Returns:
            A tuple consisting of: payload for overlay message, salt for inner message
                encryption key, nonce for the inner message encryption
        """
        future = asyncio.run_coroutine_threadsafe(
            self._overlay_channel.format_outgoing_overlay(message, researcher_id),
            self._loop
        )
        return future.result()
