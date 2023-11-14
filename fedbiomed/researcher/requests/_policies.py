# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import datetime
from typing import List, Optional, TypeVar

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
        """Implements timeout attributes"""
        super().__init__(nodes)
        self.timeout = timeout
        self._time = None

    def is_timeout(self) -> bool:
        """Returns True if timeout"""
        self._time = self._time or datetime.datetime.now()
        time_interest = datetime.datetime.now()
        return (time_interest - self._time).seconds > self.timeout

    def apply(self, requests: List[TRequest], stop: bool):
        """Applies timeout loop over requests"""

        if self._nodes:
            requests = [req for req in requests if req.node.id in self._nodes]

        for req in requests:
            if not req.has_finished() and self.is_timeout():
                req.status = RequestStatus.TIMEOUT
                if stop:
                    return self.stop(req)

        return PolicyStatus.CONTINUE


class DiscardOnTimeout(_ReplyTimeoutPolicy):
    """Discards request that does not answers in givin timeout"""
    def continue_(self, requests: TRequest) -> PolicyStatus:
        return self.apply(requests, False)


class StopOnTimeout(_ReplyTimeoutPolicy):
    """Stops the request if nodes does not answers in givin timeout"""
    def continue_(self, requests) -> PolicyStatus:
        return self.apply(requests, True)


class StopOnDisconnect(_ReplyTimeoutPolicy):
    """Stops collecting results if a node disconnects"""

    def continue_(self, requests: TRequest) -> PolicyStatus:
        """Continues federated request if nodes are not disconnect"""

        if self._nodes:
            requests = [req for req in requests if req.node.id in self._nodes]

        for req in requests:
            if req.status == RequestStatus.DISCONNECT and self.is_timeout():
                return self.stop(req)

        return PolicyStatus.CONTINUE


class StopOnError(RequestPolicy):
    """Stops collecting results if a node returns an error"""

    def continue_(self, requests: TRequest) -> PolicyStatus:
        """Continues federated request if nodes does not return error"""

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
        self.policies = policies

    def continue_all(self, requests: List[TRequest]) -> PolicyStatus:
        """Checks if reply collection should continue according to each strategy

        Returning anything different than StrategyStatus.CONTINUE stops the
        federated request loop

        Args:
            requests: List of [Request][fedbiomed.researcher.requests.Request] object
        """

        if not requests:
            return False

        status = all(
            [policy.continue_(requests=requests) == PolicyStatus.CONTINUE
                for policy in self.policies]
        )

        return PolicyStatus.CONTINUE if status else PolicyStatus.COMPLETED

    def has_stopped_any(self):
        """Returns true if request has stopped due to given strategy"""

        is_stopped = any(
            [policy.status == PolicyStatus.STOPPED for policy in self.policies]
        )

        return is_stopped

    def report(self):
        """Reports strategy stop status"""
        report = {}
        for st in self.policies:
            if st.status == PolicyStatus.STOPPED:
                report.update({st.stop_caused_by.node.id : st.__class__.__name__})

        return report
