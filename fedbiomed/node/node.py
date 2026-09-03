# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Core code of the node component.
"""

import enum
import os
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Union

from fedbiomed import __version__
from fedbiomed.common.certificate_manager import (
    CERTIFICATE_EXPIRY_WARNING_DAYS,
    CertificateManager,
    certificate_expiry,
    certificate_san_names,
    is_loopback_name,
)
from fedbiomed.common.constants import ComponentType, ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedCertificateError, FedbiomedError
from fedbiomed.common.logger import logger
from fedbiomed.common.message import (
    AdditiveSSSetupRequest,
    ApprovalRequest,
    ErrorMessage,
    FARequest,
    ListReply,
    ListRequest,
    Message,
    OverlayMessage,
    PingReply,
    PingRequest,
    PreprocRequest,
    SearchReply,
    SearchRequest,
    SecaggDeleteReply,
    SecaggDeleteRequest,
    SecaggReply,
    SecaggRequest,
    TrainingPlanStatusRequest,
    TrainRequest,
)
from fedbiomed.common.synchro import EventWaitExchange
from fedbiomed.common.tasks_queue import TasksQueue
from fedbiomed.common.utils import read_file
from fedbiomed.node.config import NodeConfig
from fedbiomed.node.dataset_manager import DatasetManager
from fedbiomed.node.history_monitor import HistoryMonitor
from fedbiomed.node.jobs import FAJob, PreprocJob
from fedbiomed.node.requests import NodeToNodeRouter
from fedbiomed.node.round import Round
from fedbiomed.node.secagg import SecaggSetup
from fedbiomed.node.secagg_manager import SecaggManager
from fedbiomed.node.training_plan_security_manager import TrainingPlanSecurityManager
from fedbiomed.transport.client import (
    ClientStatus,
    NodeClientIdentity,
    ResearcherCredentials,
)
from fedbiomed.transport.controller import GrpcController


class DiagnosticSeverity(enum.Enum):
    """How much a certificate finding matters to a node that is starting."""

    PROBLEM = "problem"
    WARNING = "warning"


@dataclass(frozen=True)
class CertificateDiagnostic:
    """One finding about the certificates a node holds."""

    severity: DiagnosticSeverity
    message: str

    def to_dict(self) -> Dict[str, str]:
        """The finding as plain values, for surfaces that serialize it."""
        return {"severity": self.severity.value, "message": self.message}


def certificate_diagnostics(config: NodeConfig) -> List[CertificateDiagnostic]:
    """What is wrong with the certificates a node holds, before it starts.

    Reports as a problem what stops the node from connecting, and as a warning
    what it tolerates but an operator should fix. `Node` raises on the problems as
    it builds its credentials, so what any surface reports here is what the node
    itself goes on to enforce.

    Without mutual authentication the node presents no identity and pins nothing,
    so nothing here stops it starting; what is registered is still reported, to be
    fixed before mutual authentication is turned on.
    """
    mutual_authentication = config.getbool(
        "authentication", "mutual_authentication", fallback="False"
    )
    blocking = (
        DiagnosticSeverity.PROBLEM
        if mutual_authentication
        else DiagnosticSeverity.WARNING
    )

    certificate_manager = CertificateManager(db_path=config.getpath("default", "db"))
    try:
        registered = certificate_manager.list()
        expiring = certificate_manager.expiring_certificates(
            CERTIFICATE_EXPIRY_WARNING_DAYS
        )
    finally:
        certificate_manager.close()

    diagnostics = []
    researcher_host = config.get("researcher", "ip")

    # A node registers at most one certificate -- its researcher's -- so what is
    # registered says on its own whether the node can pin one.
    if len(registered) > 1:
        diagnostics.append(
            CertificateDiagnostic(
                blocking,
                f"{len(registered)} certificates are registered, but a node registers "
                "at most one -- its researcher's -- so the certificate to pin is "
                "ambiguous. List them with `fedbiomed node certificate list`, and "
                "delete all but the researcher this node connects to with `fedbiomed "
                "node certificate delete`.",
            )
        )
    elif not registered and mutual_authentication:
        diagnostics.append(
            CertificateDiagnostic(
                DiagnosticSeverity.PROBLEM,
                "Mutual authentication is enabled but no researcher certificate is "
                "registered, so the node has none to pin. Register the researcher "
                "certificate with `fedbiomed node certificate register`.",
            )
        )

    for document in registered:
        component_id = document["component_id"]
        san_names = certificate_san_names(document["certificate"])

        # The node verifies the researcher under a name read from its certificate,
        # and refuses a certificate that states none.
        if not san_names:
            diagnostics.append(
                CertificateDiagnostic(
                    blocking,
                    f"The certificate registered for {component_id} states no host, so "
                    "the node cannot verify the researcher under the address it dials. "
                    "Request the researcher to reissue it for the hosts nodes reach it "
                    "at.",
                )
            )
        # Two loopback forms are the same machine, which the channel resolves for
        # itself, so they are not a mismatch.
        elif researcher_host not in san_names and not (
            is_loopback_name(researcher_host)
            and any(is_loopback_name(name) for name in san_names)
        ):
            diagnostics.append(
                CertificateDiagnostic(
                    DiagnosticSeverity.WARNING,
                    f"The certificate registered for {component_id} is issued for "
                    f"{', '.join(san_names)}, which does not include the configured "
                    f"researcher host {researcher_host}. The channel is verified under "
                    f"{san_names[0]}, the first name the certificate carries.",
                )
            )

    # An expired certificate is refused at the handshake, so it stops the node
    # rather than merely warning it.
    now = datetime.now(timezone.utc)
    for component_id, expiry in expiring:
        if expiry <= now:
            diagnostics.append(
                CertificateDiagnostic(
                    blocking,
                    f"The certificate registered for {component_id} expired on "
                    f"{expiry:%Y-%m-%d}, so the researcher's identity is refused at "
                    "the handshake. Request the researcher to renew it, and register "
                    "the new certificate here.",
                )
            )
        else:
            diagnostics.append(
                CertificateDiagnostic(
                    DiagnosticSeverity.WARNING,
                    f"The certificate registered for {component_id} expires on "
                    f"{expiry:%Y-%m-%d}. Request the researcher to renew it, and "
                    "register the new certificate here.",
                )
            )

    # The node reads both to build the identity it presents, so either one missing
    # stops it.
    if mutual_authentication:
        for label, key in (
            ("private key", "private_key"),
            ("certificate", "public_key"),
        ):
            path = config.getpath("certificate", key)
            if not os.path.isfile(path):
                diagnostics.append(
                    CertificateDiagnostic(
                        DiagnosticSeverity.PROBLEM,
                        f"The node {label} is missing from {path}, so the node cannot "
                        "present its identity under mutual authentication.",
                    )
                )

        # The researcher verifies the certificate the node presents, so the node's
        # own expiry stops it just as the researcher's does.
        certificate_path = config.getpath("certificate", "public_key")
        if os.path.isfile(certificate_path):
            try:
                expiry = certificate_expiry(read_file(certificate_path))
            except FedbiomedError as exp:
                diagnostics.append(
                    CertificateDiagnostic(
                        DiagnosticSeverity.PROBLEM,
                        f"This node's certificate at {certificate_path} could not be "
                        f"read: {exp}",
                    )
                )
            else:
                if expiry is None:
                    diagnostics.append(
                        CertificateDiagnostic(
                            DiagnosticSeverity.PROBLEM,
                            f"This node's certificate at {certificate_path} is not "
                            "readable PEM, so the node cannot present its identity "
                            "under mutual authentication. Issue a new one with "
                            "`fedbiomed node certificate generate --force`, and send "
                            "it to the researcher to register.",
                        )
                    )
                elif expiry <= now:
                    diagnostics.append(
                        CertificateDiagnostic(
                            DiagnosticSeverity.PROBLEM,
                            f"This node's certificate expired on {expiry:%Y-%m-%d}, so "
                            "the researcher refuses the connection. Issue a new one "
                            "with `fedbiomed node certificate generate --force`, and "
                            "send it to the researcher to register.",
                        )
                    )
                elif expiry <= now + timedelta(days=CERTIFICATE_EXPIRY_WARNING_DAYS):
                    diagnostics.append(
                        CertificateDiagnostic(
                            DiagnosticSeverity.WARNING,
                            f"This node's certificate expires on {expiry:%Y-%m-%d}. "
                            "Issue a new one with `fedbiomed node certificate generate "
                            "--force`, and send it to the researcher to register.",
                        )
                    )

    return diagnostics


class NodeContext:
    """What a node command acts on: the node's local state.

    Holds what the node keeps on disk -- its configuration, datasets and
    training plans -- and nothing else: what the user typed belongs to
    `NodeCLI`, running a node to `node_pm`.

    Acting on this state needs neither a running node nor the means to run
    one, so commands get this rather than a `Node`. A running node holds one
    of these too, as the local state it operates on.

    The managers are public so callers can swap them, notably for testing.
    """

    dataset_manager: DatasetManager
    tp_security_manager: TrainingPlanSecurityManager

    def __init__(self, config: NodeConfig) -> None:
        """Constructor of the class.

        Args:
            config: Configuration of the node the commands act on.
        """
        self._config = config
        self.dataset_manager = DatasetManager(
            path=self.config.getpath("default", "db"),
            min_samples=self.config.getint("security", "minimum_samples"),
        )
        self.tp_security_manager = TrainingPlanSecurityManager(
            db=self.config.getpath("default", "db"),
            node_id=self.config.get("default", "id"),
            node_name=self.config.get("default", "name"),
            hashing=self.config.get("security", "hashing_algorithm"),
            tp_approval=self.config.getbool("security", "training_plan_approval"),
        )

    @property
    def config(self) -> NodeConfig:
        """Configuration of the node the commands act on."""
        return self._config


class Node:
    """Core code of the node component.

    Defines the behaviour of the node, while communicating
    with the researcher through the `Messaging`, parsing messages from the researcher,
    either treating them instantly or queuing them,
    executing tasks requested by researcher stored in the queue.

    Owns a `NodeContext`: the local state it operates on.
    """

    def __init__(
        self,
        config: NodeConfig,
        node_args: Union[dict, None] = None,
        on_connection_state: Optional[Callable] = None,
    ):
        """Constructor of the class.

        Attributes:
            config: Node configuration
            node_args: Command line arguments for node.
            on_connection_state: Called with the state of the researcher connection
                whenever it changes. None records nothing.
        """
        self.node_args = node_args or {}
        self._on_connection_state = on_connection_state
        self._debug = bool(self.node_args.get("debug", False)) or os.environ.get(
            "FBM_DEBUG", ""
        ).lower() in ("1", "true", "yes")

        self._context = NodeContext(config)
        self._node_id = self.config.get("default", "id")
        self._node_name = self.config.get("default", "name")

        self._tasks_queue = TasksQueue(
            os.path.join(self.config.root, "var", f"queue_{self._node_id}"),
            str(os.path.join(self.config.root, "var", "tmp")),
        )

        try:
            researcher = self._researcher_credentials()
        except FedbiomedCertificateError as exp:
            # The node stops here, so the reason is recorded where the connection
            # state is read from, and not only in the logs.
            if self._on_connection_state:
                self._on_connection_state(
                    state=ClientStatus.FAILED,
                    host=self.config.get("researcher", "ip"),
                    port=self.config.get("researcher", "port"),
                    mtls=True,
                    identity_verified=False,
                    operation="mtls_startup_refused",
                    reason=str(exp),
                )
            raise

        self._grpc_client = GrpcController(
            node_id=self._node_id,
            researchers=[researcher],
            on_message=self.on_message,
            on_connection_state=on_connection_state,
        )

        self._pending_requests = EventWaitExchange(remove_delivered=True)
        self._controller_data = EventWaitExchange(remove_delivered=False)
        self._n2n_router = NodeToNodeRouter(
            self._node_id,
            self.config.getpath("default", "db"),
            self._grpc_client,
            self._pending_requests,
            self._controller_data,
        )

        # Initialize security audit logging
        logger.set_security_logs(root_path=self.config.root)

        logger.configure_security(
            component_id=self._node_id,
            component_name=ComponentType.NODE,
            fedbiomed_version=__version__,
        )

        # Log node creation
        logger.security_event(
            operation="node_initialized",
            status="success",
            researcher_id=None,
            node_name=self._node_name,
            config_path=self.config.root,
        )

    def _researcher_credentials(self) -> ResearcherCredentials:
        """Builds the researcher connection credentials.

        When mutual authentication is enabled through the `[authentication]` config
        section, the node attaches its own client identity and pins the registered
        researcher certificate. Otherwise it trusts whichever certificate the endpoint
        presents on connect.

        Returns:
            Credentials used to connect to the researcher gRPC server.

        Raises:
            FedbiomedCertificateError: mutual authentication is enabled but the
                certificates the node needs are missing or unusable.
        """
        credentials = ResearcherCredentials(
            port=self.config.get("researcher", "port"),
            host=self.config.get("researcher", "ip"),
        )

        if not self.config.getbool(
            "authentication", "mutual_authentication", fallback="False"
        ):
            return credentials

        problems = [
            diagnostic.message
            for diagnostic in certificate_diagnostics(self.config)
            if diagnostic.severity is DiagnosticSeverity.PROBLEM
        ]
        if problems:
            raise FedbiomedCertificateError(
                f"{ErrorNumbers.FB619.value}: The node cannot connect under mutual "
                "authentication: " + " ".join(problems)
            )

        credentials.mtls = True
        credentials.node_identity = self._node_identity()
        credentials.certificate = self._pinned_researcher_certificate()
        return credentials

    def _node_identity(self) -> NodeClientIdentity:
        """Loads the node's own client identity used for mutual authentication.

        Returns:
            The node's client identity (private key and certificate chain)
            presented to the researcher under mutual authentication.

        Raises:
            FedbiomedCertificateError: the node's certificate or private key
                could not be read.
        """
        try:
            private_key = read_file(self.config.getpath("certificate", "private_key"))
            certificate_chain = read_file(
                self.config.getpath("certificate", "public_key")
            )
        except FedbiomedError as exp:
            raise FedbiomedCertificateError(
                f"{ErrorNumbers.FB619.value}: Mutual authentication is enabled but "
                f"this node's certificate or private key could not be read: {exp}"
            ) from exp

        return NodeClientIdentity(
            private_key=private_key.encode("utf-8"),
            certificate_chain=certificate_chain.encode("utf-8"),
        )

    def _pinned_researcher_certificate(self) -> bytes:
        """Resolves the registered researcher server certificate to pin.

        The node connects to the single researcher declared in its `[researcher]`
        config section, and registers that researcher's certificate and no other,
        so the certificate to pin is the single registered one. That there is
        exactly one is established by `certificate_diagnostics`, which the caller
        runs first.

        Returns:
            The researcher's public server certificate, pinned under mutual
            authentication.
        """
        certificate_manager = CertificateManager(
            db_path=self.config.getpath("default", "db")
        )
        try:
            certificates = certificate_manager.list()
        finally:
            certificate_manager.close()

        return certificates[0]["certificate"].encode("utf-8")

    @property
    def node_id(self):
        """Returns id of the node"""
        return self._node_id

    @property
    def node_name(self):
        """Returns id of the node"""
        return self._node_name

    @property
    def config(self):
        """Return node config"""
        return self._context.config

    @property
    def dataset_manager(self) -> DatasetManager:
        """Manager of the node's datasets, held by its context."""
        return self._context.dataset_manager

    @dataset_manager.setter
    def dataset_manager(self, manager: DatasetManager) -> None:
        self._context.dataset_manager = manager

    @property
    def tp_security_manager(self) -> TrainingPlanSecurityManager:
        """Manager of the node's training plans, held by its context."""
        return self._context.tp_security_manager

    @tp_security_manager.setter
    def tp_security_manager(self, manager: TrainingPlanSecurityManager) -> None:
        self._context.tp_security_manager = manager

    def add_task(self, task: dict):
        """Adds a task to the pending tasks queue.

        Args:
            task: A `Message` object describing a training task
        """
        self._tasks_queue.add(task)
        # Log task added to queue
        logger.security_event(
            operation="task_queued",
            status="queued",
            researcher_id=getattr(task, "researcher_id", None),
            message_type=task.__name__,
            experiment_id=getattr(task, "experiment_id", None),
            request_id=getattr(task, "request_id", None),
        )

    def on_message(self, msg: dict):
        """Handler to be used with `Messaging` class (ie the messager).

        Called when a  message arrives through the `Messaging`.
        It reads and triggers instructions received by node from Researcher,
        mainly:
        - ping requests,
        - train requests (then a new task will be added on node's task queue),
        - search requests (for searching data in node's database).

        Args:
            msg: Incoming message from Researcher.
        """
        message: Message
        try:
            message = Message.from_dict(msg)
        except FedbiomedError as e:
            logger.error(e)  # Message was not properly formatted
            resid = msg.get("researcher_id", "unknown_researcher_id")
            self.send_error(
                ErrorNumbers.FB301,
                extra_msg="Message was not properly formatted",
                researcher_id=resid,
            )
        else:
            logger.debug(
                "Received researcher message type=%s req=%s researcher=%s experiment=%s dataset=%s round=%s",
                message.__name__,
                getattr(message, "request_id", None),
                getattr(message, "researcher_id", None),
                getattr(message, "experiment_id", None),
                getattr(message, "dataset_id", None),
                getattr(message, "round", None),
            )

            # Set security context for all logs related to this message
            with logger.security_context(
                researcher_id=getattr(message, "researcher_id", None),
                message_type=message.__name__,
                request_id=getattr(message, "request_id", None),
                experiment_id=getattr(message, "experiment_id", None),
            ):
                # Log incoming message
                logger.security_event(
                    operation="message_received_from_researcher",
                    status="received",
                )

                match message.__name__:
                    case (
                        TrainRequest.__name__
                        | SecaggRequest.__name__
                        | AdditiveSSSetupRequest.__name__
                        | FARequest.__name__
                        | PreprocRequest.__name__
                    ):
                        logger.debug(
                            "Queueing node task type=%s req=%s experiment=%s",
                            message.__name__,
                            getattr(message, "request_id", None),
                            getattr(message, "experiment_id", None),
                        )
                        self.add_task(message)
                    case SecaggDeleteRequest.__name__:
                        self._task_secagg_delete(message)
                    case OverlayMessage.__name__:
                        self._n2n_router.submit(message)
                    case SearchRequest.__name__:
                        databases = self.dataset_manager.dataset_table.search_by_tags(
                            message.tags
                        )
                        if len(databases) != 0:
                            databases = (
                                self.dataset_manager.obfuscate_private_information(
                                    databases
                                )
                            )
                        reply = SearchReply(
                            request_id=message.request_id,
                            node_id=self._node_id,
                            node_name=self._node_name,
                            researcher_id=message.researcher_id,
                            databases=databases,
                            count=len(databases),
                        )
                        self._grpc_client.send(reply)
                        # Log outgoing reply
                        logger.security_event(
                            operation="SearchReply_sent",
                            status="sent",
                        )
                    case ListRequest.__name__:
                        # Get list of all datasets
                        databases = self.dataset_manager.list_my_datasets(verbose=False)
                        databases = self.dataset_manager.obfuscate_private_information(
                            databases
                        )
                        reply = ListReply(
                            success=True,
                            request_id=message.request_id,
                            node_id=self._node_id,
                            node_name=self._node_name,
                            researcher_id=message.researcher_id,
                            databases=databases,
                            count=len(databases),
                        )
                        self._grpc_client.send(reply)
                        # Log outgoing reply
                        logger.security_event(
                            operation="ListReply_sent",
                            status="sent",
                        )

                    case PingRequest.__name__:
                        reply = PingReply(
                            request_id=message.request_id,
                            researcher_id=message.researcher_id,
                            node_id=self._node_id,
                            node_name=self._node_name,
                        )
                        self._grpc_client.send(reply)
                        # Log outgoing reply
                        logger.security_event(
                            operation="PingReply_sent",
                            status="sent",
                        )
                    case ApprovalRequest.__name__:
                        reply = self.tp_security_manager.reply_training_plan_approval_request(
                            message
                        )
                        self._grpc_client.send(reply)
                        # Log outgoing reply
                        logger.security_event(
                            operation="ApprovalReply_sent",
                            status="sent",
                        )
                    case TrainingPlanStatusRequest.__name__:
                        reply = (
                            self.tp_security_manager.reply_training_plan_status_request(
                                message
                            )
                        )
                        self._grpc_client.send(reply)
                        # Log outgoing reply
                        logger.security_event(
                            operation="TrainingPlanStatusReply_sent",
                            status="sent",
                        )
                    case _:
                        resid = msg.get("researcher_id", "unknown_researcher_id")
                        self.send_error(
                            ErrorNumbers.FB301,
                            extra_msg="This request handler is not implemented "
                            f"{message.__class__.__name__} is not implemented",
                            researcher_id=resid,
                        )

    def _task_secagg_delete(self, msg: SecaggDeleteRequest) -> None:
        """Parse a given secagg delete task message and execute secagg delete task.

        Args:
            msg: `SecaggDeleteRequest` message object to parse
        """

        secagg_id = msg.secagg_id

        reply = {
            "node_id": self._node_id,
            "node_name": self._node_name,
            "researcher_id": msg.researcher_id,
            "success": True,
            "secagg_id": secagg_id,
            "request_id": msg.request_id,
        }

        secagg_manager = SecaggManager(
            db=self.config.getpath("default", "db"), element=msg.get_param("element")
        )()

        status = secagg_manager.remove(
            secagg_id=secagg_id, experiment_id=msg.get_param("experiment_id")
        )

        if not status:
            message = (
                f"{ErrorNumbers.FB321.value}: no such secagg context "
                f"element in node database for node_id={self._node_id} "
                f"for hospital={self._node_name} "
                f"secagg_id={secagg_id}"
            )
            return self.send_error(
                extra_msg=message,
                researcher_id=msg.researcher_id,
                request_id=msg.request_id,
            )

        return self._grpc_client.send(
            SecaggDeleteReply(**{**reply, "msg": "Secagg context is deleted."})
        )

    def _task_secagg(self, request: SecaggRequest) -> None:
        """Parse a given secagg setup task message and execute secagg task.

        Args:
            request: `SecaggRequest` message object to parse
        """
        setup_arguments = request.get_dict()
        setup_arguments.pop("protocol_version")
        setup_arguments.pop("request_id")

        # Needed when using node to node communications
        setup_arguments.update(
            {
                "db": self.config.getpath("default", "db"),
                "node_id": self._node_id,
                "node_name": self._node_name,
                "n2n_router": self._n2n_router,
                "grpc_client": self._grpc_client,
                "pending_requests": self._pending_requests,
                "controller_data": self._controller_data,
            }
        )

        try:
            secagg = SecaggSetup(**setup_arguments)()
            reply: SecaggReply = secagg.setup()
        except Exception as error_message:
            error_text = repr(error_message)
            logger.error(
                "Secure aggregation setup failed on node=%s req=%s researcher=%s error=%s",
                self._node_id,
                request.request_id,
                request.researcher_id,
                error_text,
            )
            logger.debug(
                "Secure aggregation setup traceback: %s",
                traceback.format_exc(),
            )
            return self.send_error(
                request_id=request.request_id,
                researcher_id=request.researcher_id,
                extra_msg=error_text,
            )

        reply.request_id = request.request_id
        return self._grpc_client.send(reply)

    def parser_task_train(self, msg: TrainRequest) -> Union[Round, None]:
        """Parses a given training task message to create a round instance

        Args:
            msg: `TrainRequest` message object to parse

        Returns:
            a `Round` object for the training to perform, or None if no training
        """
        round_ = None
        hist_monitor = HistoryMonitor(
            node_id=self._node_id,
            node_name=self._node_name,
            experiment_id=msg.experiment_id,
            researcher_id=msg.researcher_id,
            send=self._grpc_client.send,
        )
        dataset_id = msg.get_param("dataset_id")
        data, _ = self.dataset_manager.get_dataset_entry_by_id(dataset_id)

        if data is None:
            return self.send_error(
                extra_msg=(
                    f"{ErrorNumbers.FB313.value}: Did not find proper data in local datasets "
                    f"on node={self._node_id} for dataset_id={dataset_id}"
                ),
                request_id=msg.request_id,
                researcher_id=msg.researcher_id,
                errnum=ErrorNumbers.FB313,
            )
        logger.debug(
            "Preparing training round req=%s experiment=%s round=%s dataset=%s training_plan=%s training=%s state_id=%s has_aux_var=%s",
            msg.request_id,
            msg.experiment_id,
            msg.round,
            dataset_id,
            msg.get_param("training_plan_class"),
            bool(msg.get_param("training")),
            msg.get_param("state_id"),
            msg.get_param("optim_aux_var") is not None,
        )

        dlp_and_loading_block_metadata = None
        if "dlp_id" in data:
            dlp_and_loading_block_metadata = self.dataset_manager.get_dlp_by_id(
                data["dlp_id"]
            )
        else:
            logger.debug("No data loading plan metadata for dataset=%s", dataset_id)

        round_ = Round(
            root_dir=self.config.root,
            db=self.config.getpath("default", "db"),
            node_id=self._node_id,
            node_name=self._node_name,
            training_plan=msg.get_param("training_plan"),
            training_plan_class=msg.get_param("training_plan_class"),
            model_kwargs=msg.get_param("model_args") or {},
            training_kwargs=msg.get_param("training_args") or {},
            training=msg.get_param("training") or False,
            dataset_entry=data,
            params=msg.get_param("params"),
            experiment_id=msg.get_param("experiment_id"),
            researcher_id=msg.get_param("researcher_id"),
            history_monitor=hist_monitor,
            aggregator_args=msg.get_param("aggregator_args") or None,
            node_args=self.node_args,
            tp_security_manager=self.tp_security_manager,
            round_number=msg.get_param("round"),
            dlp_and_loading_block_metadata=dlp_and_loading_block_metadata,
            aux_vars=msg.get_param("optim_aux_var"),
        )

        # the round raises an error if it cannot initialize
        try:
            err_msg = round_.initialize_arguments(msg.get_param("state_id"))
        except Exception:
            self.send_error(
                errnum=ErrorNumbers.FB300,
                extra_msg=f"{ErrorNumbers.FB300.value}: Could not initialize arguments",
                researcher_id=msg.researcher_id,
                request_id=msg.request_id,
            )
            logger.debug(
                f"Training round initialize arguments error. Details are: {traceback.format_exc()}"
            )
            return None

        if err_msg is not None:
            self.send_error(
                errnum=ErrorNumbers.FB300,
                extra_msg=(
                    f"{ErrorNumbers.FB300.value}: Could not initialize arguments for training round: {err_msg}"
                ),
                researcher_id=msg.researcher_id,
                request_id=msg.request_id,
            )
            return None

        return round_

    def task_manager(self):
        """Manages training tasks in the queue."""

        while True:
            item: Message = self._tasks_queue.get()
            # don't want to treat again in case of failure
            self._tasks_queue.task_done()

            logger.info(
                f"[TASKS QUEUE] Task received by task manager: "
                f"Researcher: {item.researcher_id} "
                f"Experiment: {item.experiment_id}"
            )

            # Set security context for all logs in this task
            with logger.security_context(
                researcher_id=item.researcher_id,
                experiment_id=item.experiment_id,
                request_id=item.request_id,
            ):
                try:
                    match item.__name__:
                        case TrainRequest.__name__:
                            round_ = self.parser_task_train(item)
                            # once task is out of queue, initiate training rounds
                            if round_ is not None:
                                # Capture start time
                                start_time = time.time()

                                # Log training round start
                                logger.security_event(
                                    operation="training_round_start",
                                    status="initiated",
                                    dataset_id=round_.dataset_entry.get("dataset_id"),
                                    training_plan_id=item.get_param(
                                        "training_plan_class"
                                    ),
                                    round_number=item.round,
                                )
                                logger.debug(
                                    "Starting node training req=%s experiment=%s round=%s dataset=%s plan=%s",
                                    item.request_id,
                                    item.experiment_id,
                                    item.round,
                                    round_.dataset_entry.get("dataset_id"),
                                    item.get_param("training_plan_class"),
                                )
                                msg = round_.run_model_training(
                                    tp_approval=self.config.getbool(
                                        "security", "training_plan_approval"
                                    ),
                                    secagg_insecure_validation=self.config.getbool(
                                        "security", "secagg_insecure_validation"
                                    ),
                                    secagg_active=self.config.getbool(
                                        "security", "secure_aggregation"
                                    ),
                                    force_secagg=self.config.getbool(
                                        "security", "force_secure_aggregation"
                                    ),
                                    secagg_arguments=item.get_param("secagg_arguments"),
                                )
                                msg.request_id = item.request_id
                                self._grpc_client.send(msg)

                                # Calculate duration
                                duration_seconds = time.time() - start_time

                                # Log training round completion
                                logger.security_event(
                                    operation="training_round_complete",
                                    status="success",
                                    dataset_id=round_.dataset_entry.get("dataset_id"),
                                    training_plan_id=item.get_param(
                                        "training_plan_class"
                                    ),
                                    round_number=item.round,
                                    duration_seconds=round(duration_seconds, 2),
                                )
                                logger.debug(
                                    "Finished node training req=%s experiment=%s round=%s reply_type=%s success=%s duration_s=%.2f",
                                    item.request_id,
                                    item.experiment_id,
                                    item.round,
                                    msg.__class__.__name__,
                                    getattr(msg, "success", None),
                                    duration_seconds,
                                )
                                del round_

                        case SecaggRequest.__name__ | AdditiveSSSetupRequest.__name__:
                            # Log secagg setup start
                            logger.security_event(
                                operation="secagg_setup_start",
                                status="initiated",
                                secagg_id=getattr(item, "secagg_id", None),
                            )
                            self._task_secagg(item)
                            # Log secagg setup complete
                            logger.security_event(
                                operation="secagg_setup_complete",
                                status="success",
                                secagg_id=getattr(item, "secagg_id", None),
                            )
                        case FARequest.__name__:
                            # Log federated analytics start
                            logger.security_event(
                                operation="federated_analytics_start",
                                status="initiated",
                            )
                            fa_job = FAJob(
                                root_dir=self.config.root,
                                dataset_manager=self.dataset_manager,
                                node_id=self._node_id,
                                node_name=self._node_name,
                                request=item,
                                allow_fa=self.config.getbool(
                                    "security", "allow_federated_analytics"
                                ),
                                db=self.config.getpath("default", "db"),
                                secagg_active=self.config.getbool(
                                    "security", "secure_aggregation"
                                ),
                                force_secagg=self.config.getbool(
                                    "security", "force_secure_aggregation"
                                ),
                                secagg_arguments=item.get_param("secagg_arguments"),
                            )
                            response = fa_job.run()
                            self._grpc_client.send(response)
                            # Log federated analytics complete
                            logger.security_event(
                                operation="federated_analytics_complete",
                                status="success",
                            )
                            del fa_job
                        case PreprocRequest.__name__:
                            # Log preprocessing start
                            logger.security_event(
                                operation="preprocessing_start",
                                status="initiated",
                            )
                            preproc_job = PreprocJob(
                                root_dir=self.config.root,
                                dataset_manager=self.dataset_manager,
                                node_id=self._node_id,
                                node_name=self._node_name,
                                request=item,
                                db=self.config.getpath("default", "db"),
                                allow_preproc=self.config.getbool(
                                    "security", "allow_preproc"
                                ),
                            )
                            response = preproc_job.run()
                            self._grpc_client.send(response)

                            # Log preprocessing complete
                            logger.security_event(
                                operation="preprocessing_complete",
                                status="success",
                            )
                            del preproc_job
                        case _:
                            errmess = (
                                f"{ErrorNumbers.FB319.value}: Undefined request message"
                            )
                            self.send_error(
                                errnum=ErrorNumbers.FB319, extra_msg=errmess
                            )

                # TODO: Test exception
                except Exception as e:
                    self.send_error(
                        request_id=item.request_id,
                        researcher_id=item.researcher_id,
                        errnum=ErrorNumbers.FB300,
                        extra_msg="Round error: " + str(e),
                    )

    def start_protocol(self) -> None:
        """Start the node to node router thread, for handling node to node message"""
        self._n2n_router.start()

    def start_messaging(self, on_finish: Optional[Callable] = None):
        """Calls the start method of messaging class.

        Args:
            on_finish: Called when the tasks for handling all known researchers have finished.
                Callable has no argument.
        """
        # Log node start
        logger.security_event(
            operation="node_started",
            status="success",
            researcher_id=None,
            node_name=self._node_name,
        )
        self._grpc_client.start(on_finish)

    def is_connected(self) -> bool:
        """Checks if node is ready for communication with researcher

        Returns:
            True if node is ready, False if node is not ready
        """
        return self._grpc_client.is_connected()

    def send_error(
        self,
        errnum: ErrorNumbers = ErrorNumbers.FB300,
        extra_msg: str = "",
        researcher_id: str = "<unknown>",
        broadcast: bool = False,
        request_id: str = None,
    ):
        """Sends an error message.

        Args:
            errnum: Code of the error.
            extra_msg: Additional human readable error message.
            researcher_id: Destination researcher.
            broadcast: Broadcast the message all available researchers
                regardless of specific researcher.
            request_id: Optional request i to reply as error to a request.
        """
        researcher_host = self.config.get("researcher", "ip")
        researcher_port = self.config.get("researcher", "port")
        try:
            connected = self.is_connected()
        except Exception:
            connected = False

        try:
            logger.debug(
                "Preparing error reply errnum=%s req=%s researcher=%s broadcast=%s connected=%s destination=%s:%s msg_len=%d",
                errnum.name,
                request_id,
                researcher_id,
                broadcast,
                connected,
                researcher_host,
                researcher_port,
                len(extra_msg),
                stack_info=True,
            )

            # Log error to console and security audit log in one call
            logger.error(
                extra_msg,
                extra={
                    "is_security": True,
                    "operation": "error_sent",
                    "status": "error",
                    "request_id": request_id,
                    "error_code": errnum.name,
                    "error_message": extra_msg,
                    "broadcast": broadcast,
                },
                researcher_id=researcher_id if researcher_id != "<unknown>" else None,
                broadcast=broadcast,
            )

            self._grpc_client.send(
                ErrorMessage(
                    request_id=request_id,
                    errnum=errnum.name,
                    node_id=self._node_id,
                    node_name=self._node_name,
                    extra_msg=extra_msg,
                    researcher_id=researcher_id,
                ),
                broadcast=broadcast,
            )

            logger.debug(
                "Error reply dispatched errnum=%s req=%s researcher=%s broadcast=%s connected=%s",
                errnum.name,
                request_id,
                researcher_id,
                broadcast,
                connected,
            )
        except Exception as e:
            logger.error(
                f"{ErrorNumbers.FB601.value}: Cannot send error message: {e}",
                exc_info=True,
            )
