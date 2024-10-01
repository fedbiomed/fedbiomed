import copy
import os
import tempfile
import unittest
from json import decoder
from typing import Any, Dict
from unittest.mock import ANY, MagicMock, patch

import torch
from testsupport import fake_training_plan
#############################################################
# Import NodeTestCase before importing FedBioMed Module
from testsupport.base_case import NodeTestCase
from testsupport.fake_secagg_manager import FakeSecaggServkeyManager
# import dummy classes
from testsupport.fake_uuid import FakeUuid

from fedbiomed.common.constants import (ErrorNumbers, SecaggElementTypes,
                                        TrainingPlans,
                                        __messaging_protocol_version__,
                                        _BaseEnum)
from fedbiomed.common.message import (ApprovalRequest, ErrorMessage,
                                      ListRequest, Message, PingReply,
                                      PingRequest, SearchRequest,
                                      SecaggDeleteReply, SecaggDeleteRequest,
                                      SecaggReply, SecaggRequest,
                                      TrainingPlanStatusRequest, TrainRequest)
from fedbiomed.common.models import TorchModel
from fedbiomed.common.optimizers.declearn import (RidgeRegularizer,
                                                  ScaffoldClientModule,
                                                  YogiModule)
from fedbiomed.common.optimizers.generic_optimizers import DeclearnOptimizer
from fedbiomed.common.optimizers.optimizer import Optimizer
from fedbiomed.common.serializer import Serializer
from fedbiomed.node.dataset_manager import DatasetManager
from fedbiomed.node.environ import environ
from fedbiomed.node.history_monitor import HistoryMonitor

from fedbiomed.node.node import Node
from fedbiomed.node.round import Round

#############################################################


class TestNode(NodeTestCase):

    train_request = TrainRequest(
        researcher_id="researcher-id",
        experiment_id="experiment-1",
        training_plan="x",
        dataset_id="dataset_id_1234",
        training_args={},
        model_args={},
        training=True,
        training_plan_class="MyTrainingPlan",
        aggregator_args={},
        state_id=None,
        params={"x": 1},
        round=1,
    )

    secagg_request = SecaggRequest(
        researcher_id="r1",
        experiment_id="experiment-id",
        secagg_id="secagg-id",
        parties=["n1", "n2"],
        element=0,
    )

    ping_request = PingRequest(researcher_id="researcher_id")

    list_request = ListRequest(researcher_id="researcher-id")

    search_request = SearchRequest(researcher_id="researcher-id", tags=["data"])

    approval_request = ApprovalRequest(
        researcher_id="researcher-id",
        description="hmmm",
        training_plan="class MM:;pass",
    )
    tp_status_request = TrainingPlanStatusRequest(
        researcher_id="researcher-id",
        experiment_id="experiment-id",
        training_plan="class MM:;pass",
    )

    secagg_delete_request = SecaggDeleteRequest(
        researcher_id="researcher_id_1234",
        secagg_id="my_test_secagg_id",
        element=0,
        experiment_id="a_dummy_experiment_id",
    )

    @classmethod
    def setUpClass(cls):
        # Important to instantiate fake environ
        super().setUpClass()

        # --------------------------------------

    def setUp(self):
        """Sets up objects for unit tests"""

        self.database_val = [
            {
                "database_id": "1234",
                "path": "/path/to/my/dataset",
                "name": "test_dataset",
            }
        ]
        self.database_list = [
            {
                "database_id": "1234",
                "path": "/path/to/my/dataset",
                "name": "test_dataset1",
            },
            {
                "database_id": "5678",
                "path": "/path/to/another/dataset",
                "name": "test_dataset2",
            },
        ]

        self.database_id = {
            "database_id": "1234",
            "path": "/path/to/my/dataset",
            "name": "test_dataset1",
        }

        # patchers
        self.grpc_controller_patch = patch(
            "fedbiomed.transport.controller.GrpcController.__init__",
            autospec=True,
            return_value=None,
        )
        self.grpc_send_patch = patch(
            "fedbiomed.transport.controller.GrpcController.send", autospec=True
        )

        self.grpc_controller_patcher = self.grpc_controller_patch.start()
        self.grpc_send_mock = self.grpc_send_patch.start()

        self.task_queue_patch = patch(
            "fedbiomed.common.tasks_queue.TasksQueue.__init__",
            autospec=True,
            return_value=None,
        )
        self.task_patcher = self.task_queue_patch.start()

        self.exchange_patch = patch(
            "fedbiomed.node.node.EventWaitExchange", autospec=True
        )
        self.exchange_patcher = self.exchange_patch.start()

        self.n2n_router_patch = patch(
            "fedbiomed.node.node.NodeToNodeRouter", autospec=True
        )
        self.n2n_router_patcher = self.n2n_router_patch.start()

        # mocks
        mock_dataset_manager = DatasetManager()
        mock_dataset_manager.search_by_tags = MagicMock(return_value=self.database_val)
        mock_dataset_manager.list_my_data = MagicMock(return_value=self.database_list)
        mock_model_manager = MagicMock()
        mock_dataset_manager.reply_training_plan_status_request = MagicMock(
            return_value=None
        )
        mock_dataset_manager.get_by_id = MagicMock(return_value=self.database_id)

        self.model_manager_mock = mock_model_manager

        # creating Node objects
        self.n1 = Node(mock_dataset_manager, mock_model_manager)
        self.n2 = Node(mock_dataset_manager, mock_model_manager)

    def tearDown(self) -> None:
        # stopping patches
        self.grpc_send_patch.stop()
        self.task_queue_patch.stop()
        self.grpc_controller_patch.stop()
        self.exchange_patch.stop()
        self.n2n_router_patch.stop()

    @patch("fedbiomed.common.tasks_queue.TasksQueue.add")
    def test_node_01_add_task_normal_case_scenario(self, task_queue_add_patcher):
        """Tests add_task method (in the normal case scenario)"""

        self.n1.add_task(self.train_request)
        task_queue_add_patcher.assert_called_once_with(self.train_request)
        task_queue_add_patcher.reset_mock()

    @patch("fedbiomed.common.tasks_queue.TasksQueue.add")
    def test_node_02_on_message_normal_case_scenario_train_secagg_reply(
        self,
        task_queue_add,
    ):
        """Tests `on_message` method (normal case scenario), with train/secagg command"""
        # test 1: test normal case scenario, where `command` = 'train' or 'secagg'

        for message in [
            self.train_request.to_dict(),
            self.secagg_request.to_dict(),
        ]:
            # action
            self.n1.on_message(message)

            # checks
            task_queue_add.assert_called_once_with(Message.from_dict(message))
            task_queue_add.reset_mock()

    def test_node_03_on_message_normal_case_scenario_ping(
        self,
    ):
        """Tests `on_message` method (normal case scenario), with ping command"""

        # action
        self.n1.on_message(self.ping_request.to_dict())
        self.grpc_send_mock.assert_called_once()

    @patch("fedbiomed.node.node.SecaggManager")
    def test_node_04_on_message_normal_case_scenario_secagg_delete(self, skm):
        """Tests `on_message` method (normal case scenario), with secagg-delete command"""

        skm.return_value.return_value.remove.return_value = True
        self.n1.on_message(self.secagg_delete_request.to_dict())
        self.grpc_send_mock.assert_called_once()

    def test_node_05_on_message_normal_case_scenario_search(self):
        """Tests `on_message` method (normal case scenario), with search command"""
        # action
        self.n1.on_message(self.search_request.to_dict())
        self.grpc_send_mock.assert_called_once()

    def test_node_06_on_message_normal_case_scenario_list(self):
        """Tests `on_message` method (normal case scenario), with list command"""

        # action
        self.n1.on_message(self.list_request.to_dict())
        self.grpc_send_mock.assert_called_once()

    def test_node_07_on_message_normal_case_scenario_model_status(
        self,
    ):
        """Tests normal case scenario, if command is equals to 'training-plan-status"""

        self.n1.on_message(self.tp_status_request.to_dict())
        self.model_manager_mock.reply_training_plan_status_request.assert_called_once_with(
            self.tp_status_request
        )

    def test_node_08_on_message_unknown_command(self):
        """Tests Exception is handled if command is not a known command
        (in `on_message` method)"""
        ping_reply = PingReply(researcher_id="r1", node_id="n1")

        # action
        self.n1.on_message(ping_reply.to_dict())
        error = self.grpc_send_mock.call_args.args[1]
        self.assertIsInstance(error, ErrorMessage)

    def test_node_11_on_message_fail_msg_not_deserializable(self):
        """Tests case where a error raised (because unable to deserialize message)"""
        # Not desearializable
        ping_msg = {"researcher_id": "re1", "request_id": "1234"}

        self.n1.on_message(ping_msg)

        error = self.grpc_send_mock.call_args.args[1]
        self.assertIsInstance(error, ErrorMessage)

    @patch("fedbiomed.node.node.Round", autospec=True)
    @patch("fedbiomed.node.history_monitor.HistoryMonitor.__init__", spec=True)
    def test_node_12_parser_task_train_create_round(
        self,
        history_monitor_patch,
        round_patch,
    ):
        """Tests if rounds are created accordingly - running normal case scenario
        (in `parser_task_train` method)"""

        history_monitor_patch.spec = True
        history_monitor_patch.return_value = None
        round_patch.return_value.initialize_arguments.return_value = None

        round_ = self.n1.parser_task_train(self.train_request)
        self.assertIsInstance(round_, Round)
        round_patch.assert_called_once()

    @patch("fedbiomed.node.history_monitor.HistoryMonitor.__init__")
    @patch("fedbiomed.node.round.Round.__init__")
    def test_node_13_parser_task_train_no_dataset_found(
        self,
        round_init,
        history_monitor_patch,
    ):
        """Tests parser_task_train method, case where no dataset has been found"""
        # defining patchers
        history_monitor_patch.return_value = None
        round_init.return_value = None

        mock_dataset_manager = MagicMock()
        mock_dataset_manager.get_by_id = MagicMock(return_value=None)
        self.n1.dataset_manager = mock_dataset_manager
        self.n1.parser_task_train(self.train_request)

        error = self.grpc_send_mock.call_args.args[1]
        self.assertIsInstance(error, ErrorMessage)

    @patch("fedbiomed.node.node.Round", autospec=True)
    @patch("fedbiomed.node.history_monitor.HistoryMonitor.__init__", spec=True)
    def test_node_14_parser_task_train_create_round_deserializer_str_msg(
        self, history_monitor_patch, round_patch
    ):
        """Tests if message is correctly deserialized if message is in string"""

        # defining arguments
        dict_msg_1_dataset = {
            "protocol_version": str(__messaging_protocol_version__),
            "model_args": {"lr": 0.1},
            "training_args": {"some_value": 1234},
            "training_plan": "TP",
            "training_plan_class": "my_test_training_plan",
            "params": {"x": 0},
            "experiment_id": "experiment_id_1234",
            "state_id": None,
            "secagg_arguments": {
                "secagg_servkey_id": None,
                "secagg_random": None,
                "secagg_clipping_range": None,
            },
            "round": 1,
            'researcher_id': 'researcher_id_1234',
            'dataset_id': 'dataset_id_1234',
            'training': True,
            'aggregator_args': {},
            "optim_aux_var": None,
        }
        # we convert this dataset into a string
        msg1_dataset = TrainRequest(**dict_msg_1_dataset)
        round_patch.return_value.initialize_arguments.return_value = None

        # defining patchers

        history_monitor_patch.spec = True
        history_monitor_patch.return_value = None

        # action
        self.n1.parser_task_train(msg1_dataset)

        # checks
        round_patch.assert_called_once_with(
            model_kwargs=dict_msg_1_dataset['model_args'],
            training_kwargs=dict_msg_1_dataset['training_args'],
            training=True,
            dataset=self.database_id,
            params=dict_msg_1_dataset['params'],
            experiment_id=dict_msg_1_dataset['experiment_id'],
            researcher_id=dict_msg_1_dataset['researcher_id'],
            history_monitor=unittest.mock.ANY,
            aggregator_args=None,
            node_args=None,
            training_plan=dict_msg_1_dataset['training_plan'],
            training_plan_class=dict_msg_1_dataset['training_plan_class'],
            round_number=1,
            dlp_and_loading_block_metadata=None,
            aux_vars=dict_msg_1_dataset['optim_aux_var']
        )

    @patch("fedbiomed.node.node.Round", autospec=True)
    @patch("fedbiomed.node.history_monitor.HistoryMonitor.__init__", spec=True)
    def test_node_15_parser_task_train_create_round_deserializer_bytes_msg(
        self, history_monitor_patch, round_patch
    ):
        """Tests if message is correctly deserialized if message is in bytes"""

        # defining arguments
        dict_msg_1_dataset = {
            "protocol_version": str(__messaging_protocol_version__),
            "model_args": {"lr": 0.1},
            "training_args": {"some_value": 1234},
            "training": True,
            "training_plan": "TP",
            "training_plan_class": "my_test_training_plan",
            "params": {"x": 0},
            "experiment_id": "experiment_id_1234",
            "state_id": None,
            "researcher_id": "researcher_id_1234",
            "secagg_arguments": None,
            "dataset_id": "dataset_id_1234",
            "round": 0,
            'aggregator_args': {},
            "optim_aux_var": None
        }

        #
        msg_1_dataset = TrainRequest(**dict_msg_1_dataset)

        # defining patchers

        history_monitor_patch.spec = True
        history_monitor_patch.return_value = None
        round_patch.return_value.initialize_arguments.return_value = None

        # action
        self.n1.parser_task_train(msg_1_dataset)

        # checks
        round_patch.assert_called_once_with(
            training_plan=dict_msg_1_dataset['training_plan'],
            training_plan_class=dict_msg_1_dataset['training_plan_class'],
            model_kwargs=dict_msg_1_dataset['model_args'],
            training_kwargs=dict_msg_1_dataset['training_args'],
            training=True,
            dataset=self.database_id,
            params=dict_msg_1_dataset['params'],
            experiment_id=dict_msg_1_dataset['experiment_id'],
            researcher_id=dict_msg_1_dataset['researcher_id'],
            history_monitor=unittest.mock.ANY,
            node_args=None,
            aggregator_args=None,
            round_number=0,
            dlp_and_loading_block_metadata=None,
            aux_vars=dict_msg_1_dataset['optim_aux_var']
        )

    @patch("fedbiomed.common.tasks_queue.TasksQueue.get")
    def test_node_16_task_manager_exception_raised_task_queue(
        self, tasks_queue_get_patch
    ):
        """Simulates an Exception (SystemError) triggered by `tasks_queue.get`"""
        # defining patchers
        tasks_queue_get_patch.side_effect = SystemError(
            "mimicking an exception coming from "
        )

        # action
        with self.assertRaises(SystemError):
            # checks if `SystemError` is caught (triggered by patched `tasks_queue.get`)
            self.n1.task_manager()

    @patch("fedbiomed.node.secagg._secagg_setups.SKManager")
    @patch("fedbiomed.common.tasks_queue.TasksQueue.task_done")
    @patch("fedbiomed.node.node.Node._task_secagg")
    @patch("fedbiomed.common.tasks_queue.TasksQueue.get")
    def test_node_19_task_manager_secagg_exception_raised_task_done(
        self,
        tasks_queue_get_patch,
        task_secagg_patch,
        tasks_queue_task_done_patch,
        patch_servkey_manager,
    ):
        """Tests if an Exception (SystemExit) is triggered when calling
        `TasksQueue.task_done` method for secagg message"""
        # defining patchers
        tasks_queue_get_patch.return_value = {
            "protocol_version": "99.99",
            "researcher_id": "my_test_researcher",
            "secagg_id": "my_test_secagg",
            "element": 33,
            "experiment_id": "my_experiment",
            "parties": [],
        }
        task_secagg_patch.return_value = None
        self.grpc_send_mock.return_value = None

        tasks_queue_task_done_patch.side_effect = SystemExit(
            "Mimicking an exception happening in" + "`TasksQueue.task_done` method"
        )  # noqa

        patch_servkey_manager.return_value = FakeSecaggServkeyManager()

        # action
        with self.assertRaises(SystemExit):
            # checks if `SystemExit` is raised (should be triggered by `TasksQueue.task_done`)
            self.n1.task_manager()

        # check that `Messaging.send_message` has not been called
        self.assertEqual(self.grpc_send_mock.call_count, 0)

    @patch("fedbiomed.transport.controller.GrpcController.start")
    def test_node_20_start_messaging_normal_case_scenario(self, msg_start_patch):
        """Tests `start_messaging` method (normal_case_scenario)"""
        # arguments
        block = True

        # action
        self.n1.start_messaging(block)

        # checks
        msg_start_patch.assert_called_once_with(block)

    def test_node_21_send_error_normal_case_scenario(self):
        """Tests `send_error` method (normal case scenario)"""
        # arguments
        errnum = ErrorNumbers.FB100
        extra_msg = "this is a test_send_error"
        researcher_id = "researcher_id_1224"

        # action
        self.n1.send_error(errnum, extra_msg, researcher_id)

        # checks
        self.grpc_send_mock.assert_called_once()

    def test_node_22_on_message_search_privacy_obfuscation(self):
        """Tests that privacy-sensitive information is not revealed (here path files)"""

        databases = [
            dict(
                data_type="medical-folder",
                dataset_parameters={
                    "tabular_file": "path/to/tabular/file",
                    "index_col": 0,
                },
                **self.database_val[0],
            )
        ]

        dataset_manager = DatasetManager()
        dataset_manager.search_by_tags = MagicMock(return_value=databases)
        n3 = Node(dataset_manager, self.model_manager_mock)
        n3.on_message(self.search_request.to_dict())

        # check privacy-sensitive info a case-by-case basis
        database_info = self.grpc_send_mock.call_args[0][1].get_param("databases")[0]
        self.assertNotIn("path", database_info)
        self.assertNotIn("tabular_file", database_info["dataset_parameters"])

    @patch("fedbiomed.node.node.SecaggSetup")
    def test_node_23_task_secagg(self, secagg_setup):
        """Tests `_task_secagg` normal (successful) case"""
        # Test .setup()execution. It is normal the get result as success False since setup will fail
        # due to not existing certificate files
        x = MagicMock()
        secagg_setup.return_value.return_value = x
        x.setup.return_value = MagicMock()
        self.n1._task_secagg(self.secagg_request)

        # Test setup error case ---------------------------------------------------------------
        x.setup.side_effect = Exception
        self.n1._task_secagg(self.secagg_request)
        error = self.grpc_send_mock.call_args.args[1]
        self.assertIsInstance(error, ErrorMessage)

    def test_node_24_task_secagg_delete(self):
        """Tests `_task_secagg` with bad message values"""

        # Bad element type --------------------------------------------------------------------------
        req = {
            "protocol_version": str(__messaging_protocol_version__),
            "researcher_id": "party1",
            "secagg_id": "my_dummy_secagg_id",
            "request_id": "request",
            "element": 12,
            "experiment_id": "my_test_experiment",
        }
        # Test remove status ----------------------------------------------------------------
        # status will be false since there is no registry in DB
        req["element"] = 0
        request = SecaggDeleteRequest(**req)
        self.n1._task_secagg_delete(request)
        error = self.grpc_send_mock.call_args.args[1]
        self.assertIsInstance(error, ErrorMessage)
        self.grpc_send_mock.reset_mock()

        # # Test raising error
        with patch("fedbiomed.node.node.SecaggManager") as skm:
            skm.return_value.return_value.remove.return_value = False
            req["element"] = 0
            request = SecaggDeleteRequest(**req)
            self.n1._task_secagg_delete(request)
            error = self.grpc_send_mock.call_args.args[1]
            self.assertIsInstance(error, ErrorMessage)
            self.grpc_send_mock.reset_mock()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
