# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import datetime
from typing import List, Optional, TypeVar, Dict

from ._status import PolicyStatus, RequestStatus


TRequest = TypeVar('TRequest')


class RequestPolicy:
    """Base strategy to collect replies from remote agents"""

    def __init__(self, nodes: Optional[List[str]] = None):
        self.status = None
        self._nodes = nodes
        self.stop_caused_by = None

    def continue_(self, requests) -> PolicyStatus:
        """Default strategy stops collecting result once all nodes has answered

        Returns:
            False stops the iteration
        """

        has_finished = all([req.has_finished() for req in requests])
        return self.keep() if not has_finished else self.completed()

    def stop(self, req) -> PolicyStatus:
        """Stop sign for strategy"""
        self.status = PolicyStatus.STOPPED
        self.stop_caused_by = req

        return PolicyStatus.STOPPED

    def keep(self) -> PolicyStatus:
        """Keeps continue collecting replies from nodes"""
        self.status = PolicyStatus.CONTINUE

        return PolicyStatus.CONTINUE

    def completed(self) -> PolicyStatus:
        """Updates status of strategy as completed without any issue"""
        self.status = PolicyStatus.COMPLETED

        return PolicyStatus.COMPLETED


class _ReplyTimeoutPolicy(RequestPolicy):
    """Base class for Timeout policies"""

    def __init__(self, timeout: int = 5, nodes: Optional[List[str]] = None) -> None:
        """Implements timeout attributes

        Args:
            timeout: maximum time for policy
            nodes: optional list of nodes to apply the policy. By default applies to all known nodes of request.
        """
        super().__init__(nodes)
        self.timeout = timeout
        self._time = None

    def is_timeout(self) -> bool:
        """Checks if timeout is reached.

        Returns:
            True if timeout
        """
        self._time = self._time or datetime.datetime.now()
        time_interest = datetime.datetime.now()
        return (time_interest - self._time).seconds > self.timeout

    def apply(self, requests: List[TRequest], stop: bool) -> PolicyStatus:
        """Applies timeout loop over requests

        Args:
            requests: list of requests to check for timeout
            stop: whether to fail (STOPPED) if some request has reached the timeout

        Returns:
            CONTINUE if no node has reached the timeout, or if `stop` is False
        """

        if self._nodes:
            requests = [req for req in requests if req.node.id in self._nodes]

        for req in requests:
            is_timeout = self.is_timeout()
            if not req.has_finished() and is_timeout:
                req.status = RequestStatus.TIMEOUT
                if stop:
                    return self.stop(req)

        return PolicyStatus.CONTINUE


class DiscardOnTimeout(_ReplyTimeoutPolicy):
    """Discards request that do not answer in given timeout"""
    def continue_(self, requests: TRequest) -> PolicyStatus:
        """Discards requests that reach timeout, always continue

        Returns:
            CONTINUE
        """
        return self.apply(requests, False)


class StopOnTimeout(_ReplyTimeoutPolicy):
    """Stops the request if nodes do not answer in given timeout"""
    def continue_(self, requests) -> PolicyStatus:
        """Continues federated request if nodes dont reach timeout

        Returns:
            CONTINUE if no node reached timeout, STOPPED if some node reached timeout
                and timeout is reached
        """
        return self.apply(requests, True)


class StopOnDisconnect(_ReplyTimeoutPolicy):
    """Stops collecting results if a node disconnects
    """

    def continue_(self, requests: TRequest) -> PolicyStatus:
        """Continues federated request if nodes are not disconnect

        Returns:
            CONTINUE if no node disconnect found, STOPPED if some node disconnect found
                and timeout is reached
        """

        if self._nodes:
            requests = [req for req in requests if req.node.id in self._nodes]

        for req in requests:
            is_timeout = self.is_timeout()
            if req.status == RequestStatus.DISCONNECT and is_timeout:
                return self.stop(req)

        return PolicyStatus.CONTINUE


class StopOnError(RequestPolicy):
    """Stops collecting results if a node returns an error
    """

    def continue_(self, requests: TRequest) -> PolicyStatus:
        """Continues federated request if nodes does not return error

        Returns:
            CONTINUE if no error found, STOPPED if some error found
        """

        if self._nodes:
            requests = [req for req in requests if req.node.id in self._nodes]

        for req in requests:
            if req.error:
                return self.stop(req)

        return PolicyStatus.CONTINUE


class PolicyController:

    def __init__(
        self,
        policies: Optional[List[RequestPolicy]] = None,
    ):

        policies = policies or []
        policies.insert(0, RequestPolicy())
        self._policies = policies

    def continue_all(self, requests: List[TRequest]) -> PolicyStatus:
        """Checks if all policies indicate to continue.

        Args:
            requests: List of [Request][fedbiomed.researcher.requests.Request] objects to
                check against policies

        Returns:
            CONTINUE if all policies indicates to continue
        """

        if not requests:
            return False

        status = all(
            [policy.continue_(requests=requests) == PolicyStatus.CONTINUE
                for policy in self._policies]
        )

        return PolicyStatus.CONTINUE if status else PolicyStatus.COMPLETED

    def has_stopped_any(self) -> bool:
        """Checks if any of the policies indicates to stop

        Returns:
            True if request has stopped due to given strategy
        """

        is_stopped = any(
            [policy.status == PolicyStatus.STOPPED for policy in self._policies]
        )

        return is_stopped

    def report(self) -> Dict[str, str]:
        """Reports strategy stop status

        Returns:
            Dict of policies stopped, indexed by the node ID that caused the stop
        """
        report = {}
        for st in self._policies:
            if st.status == PolicyStatus.STOPPED:
                report.update({st.stop_caused_by.node.id : st.__class__.__name__})

        return report
