import unittest
from unittest.mock import MagicMock, patch

from testsupport.base_mocks import MockRequestModule
from testsupport.fake_researcher_secagg import FakeSecAgg

import fedbiomed
from fedbiomed.common.constants import __breakpoints_version__, SecureAggregationSchemes
from fedbiomed.common.exceptions import FedbiomedSecureAggregationError
from fedbiomed.researcher.datasets import FederatedDataSet
from fedbiomed.researcher.federated_workflows import FederatedWorkflow
from fedbiomed.researcher.secagg import SecureAggregation


class TestFederatedWorkflow(unittest.TestCase, MockRequestModule):
    def setUp(self):
        MockRequestModule.setUp(
            self,
            module="fedbiomed.researcher.federated_workflows._federated_workflow.Requests",
        )
        super().setUp()
        self.abstract_methods_patcher = patch.multiple(
            FederatedWorkflow, __abstractmethods__=set()
        )
        self.abstract_methods_patcher.start()

    def tearDown(self):
        super().tearDown()
        self.abstract_methods_patcher.stop()

    def test_federated_workflow_01_initialization(self):
        """Test initialization of federated workflow, only cases where correct parameters are provided"""
        # FederatedWorkflow must be default-constructible
        exp = FederatedWorkflow()
        self.assertIsNone(exp.tags())  # by default, tags set to None
        self.assertIsNone(exp.nodes())  # by default, nodes set to None
        self.assertIsNone(
            exp.training_data()
        )  # by default, training data is initialized to something
        self.assertIsNotNone(
            exp.experimentation_folder()
        )  # by default, exp folder is initialized to something
        # SecAgg
        self.assertTrue(
            isinstance(exp.secagg, SecureAggregation)
        )  # set to inactive SecureAggregation
        self.assertFalse(exp.secagg.active)

        # Test all possible combinations of init arguments
        _training_data = MagicMock(spec=fedbiomed.researcher.datasets.FederatedDataSet)
        _secagg = MagicMock(spec=fedbiomed.researcher.secagg.SecureAggregation)
        parameters_and_possible_values = {
            "tags": (None, None, ["one-tag", "another-tag"]),
            "nodes": (["one-node"], None, None),
            "training_data": (
                _training_data,
                {"one-node": {"tags": ["one-tag"]}},
                None,
            ),
            "experimentation_folder": ("folder_name", None, None),
            "secagg": (True, False, _secagg),
            "save_breakpoints": (True, False, True),
        }
        # Compute cartesian product of parameter values to obtain all possible combinations

        combs = [
            {key: value[i] for key, value in parameters_and_possible_values.items()}
            for i in range(3)
        ]

        for params in combs:
            try:
                exp = FederatedWorkflow(**params)
            except Exception as e:
                print(f"Exception {e} raised with the following parameters {params}")
                raise e

        # Special corner cases that deserve additional testing
        # Test case where tags are None but we are setting training data
        _training_data.node_ids.return_value = [
            "alice",
            "bob",
        ]  # make sure that nodes can be correctly inferred
        exp = FederatedWorkflow(
            nodes=["alice", "bob"],
            training_data=_training_data,
            secagg=True,
            save_breakpoints=True,
        )
        self.assertListEqual(exp.nodes(), ["alice", "bob"])
        self.assertEqual(exp.training_data(), _training_data)
        self.assertTrue(isinstance(exp.secagg, SecureAggregation))
        self.assertTrue(exp.secagg.active)
        self.assertTrue(exp.save_breakpoints())
        # Test special cases regarding training data:
        # a. when tags are provided but training data is not provided, build training data from tags
        self.fake_search_reply = {
            "node1": [{"my-metadata": "is-the-best", "tags": ["some-tags"]}]
        }
        self.mock_requests.return_value.search.return_value = self.fake_search_reply
        exp = FederatedWorkflow(tags="some-tags")
        self.assertListEqual(exp.tags(), ["some-tags"])
        self.assertDictEqual(exp.training_data().data(), self.fake_search_reply)

        # b. when tags, nodes and training data are provided, the latter takes precedence and tags are set to None
        with self.assertRaises(SystemExit):
            exp = FederatedWorkflow(
                tags="some-tags", nodes=["wrong", "nodes"], training_data=_training_data
            )

    def test_federated_workflow_02_set_tags(self):
        exp = FederatedWorkflow()

        exp.set_tags("just-a-str")
        self.assertEqual(exp.tags(), ["just-a-str"])

        exp.set_tags(["first", "second"])
        self.assertEqual(exp.tags(), ["first", "second"])

        # Test invalid type and values
        with self.assertRaises(SystemExit):  # FedbiomedValueError,
            exp.set_tags(None)

        with self.assertRaises(SystemExit):  # FedbiomedValueError
            exp.set_tags([])

        with self.assertRaises(SystemExit):  # FedbiomedTypeError
            exp.set_tags(15)

        with self.assertRaises(SystemExit):
            exp.set_tags(["x", "y", 15])

    def test_federated_workflow_03_set_nodes(self):
        exp = FederatedWorkflow()
        exp.set_nodes(None)

        self.assertIsNone(exp.nodes())
        exp.set_nodes(["first", "second"])
        self.assertEqual(exp.nodes(), ["first", "second"])

        # Invalid arguments
        with self.assertRaises(SystemExit):
            exp.set_nodes(["node-1", "node-2", 15])

        with self.assertRaises(SystemExit):
            exp.set_nodes("invalid_type")

    def test_federated_workflow_04_set_training_data(self):
        exp = FederatedWorkflow()

        with self.assertRaises(SystemExit):
            exp.set_training_data(None, from_tags=False)

        # Invalid from_tags argument
        with self.assertRaises(SystemExit):
            exp.set_training_data(None, from_tags="invalid-type")

        with self.assertRaises(SystemExit):
            exp.set_training_data("not-none", from_tags=True)

        # Invalid training_data argument
        with self.assertRaises(SystemExit):
            exp.set_training_data("not-none", from_tags=False)

        self.assertIsNone(exp.training_data())

        self.fake_search_reply = {
            "node1": [{"my-metadata": "is-the-best", "tags": ["some-tag"]}]
        }
        self.mock_requests.return_value.search.return_value = self.fake_search_reply
        exp.set_tags("just-a-str")
        exp.set_training_data(None, from_tags=True)
        self.assertDictEqual(
            exp.training_data().data(),
            {"node1": {"my-metadata": "is-the-best", "tags": ["some-tag"]}},
        )
        _training_data = MagicMock(spec=fedbiomed.researcher.datasets.FederatedDataSet)
        exp.set_training_data(_training_data)
        self.assertEqual(exp.training_data(), _training_data)

    def test_federated_workflow_05_set_experimentation_folder(self):
        exp = FederatedWorkflow()
        with patch(
            "fedbiomed.researcher.federated_workflows"
            "._federated_workflow.create_exp_folder"
        ) as mock_exp_folder_creat:
            exp.set_experimentation_folder()
            mock_exp_folder_creat.assert_called_once_with(
                exp.config.vars["EXPERIMENTS_DIR"]
            )
            self.assertIsNotNone(exp.experimentation_folder())
            old_folder = exp.experimentation_folder()
            mock_exp_folder_creat.reset_mock()
            exp.set_experimentation_folder("new-name")
            mock_exp_folder_creat.assert_called_once_with(
                exp.config.vars["EXPERIMENTS_DIR"], "new-name"
            )

            mock_exp_folder_creat.side_effect = lambda x: x
            # Invalid argument type
            with self.assertRaises(SystemExit):
                result = exp.set_experimentation_folder(15)

    def test_federated_workflow_07_set_secagg(self):
        exp = FederatedWorkflow()
        exp.set_secagg(True)
        self.assertTrue(isinstance(exp.secagg, SecureAggregation))
        self.assertTrue(exp.secagg.active)
        _secagg = MagicMock(spec=fedbiomed.researcher.secagg.SecureAggregation)
        exp.set_secagg(_secagg)
        self.assertEqual(exp.secagg, _secagg)

        with self.assertRaises(SystemExit):
            exp.set_secagg("invalid")

        bad_schemes = [False, 3, None, "scheme", [SecureAggregationSchemes.LOM]]
        for scheme in bad_schemes:
            with self.assertRaises(SystemExit):
                exp.set_secagg(True, scheme)

    def test_federated_workflow_08_consistency_fds_tags(self):
        self.fake_search_reply = {
            "node1": [{"my-metadata": "is-the-best", "tags": ["some-tags"]}]
        }
        self.mock_requests.return_value.search.return_value = self.fake_search_reply
        exp = FederatedWorkflow()
        # setting tags when training data is None -> simply set tags
        exp.set_tags(["some-tags"])
        self.assertListEqual(exp.tags(), ["some-tags"])
        self.assertDictEqual(exp.training_data().data(), self.fake_search_reply)
        self.assertIsNone(exp.nodes())  # no filtering applied

        # resetting tags to None when training data is not None -> simply set tags to None
        exp._tags = None
        exp.set_training_data(FederatedDataSet(self.fake_search_reply))
        self.assertIsNone(exp.tags())
        self.assertDictEqual(exp.training_data().data(), self.fake_search_reply)

        # setting training data from tags, when tags is None -> raise error
        with self.assertRaises(SystemExit):
            exp.set_training_data(None, from_tags=True)

        # set tags when training data is not None -> reset training data based on new tags
        exp.set_training_data(FederatedDataSet(self.fake_search_reply))
        self.fake_search_reply = {
            "node2": [{"my-metadata": "is-the-bestest", "tags": ["other-tags"]}]
        }
        self.mock_requests.reset_mock()
        self.mock_requests.return_value.search.return_value = self.fake_search_reply
        exp.set_tags("other-tags")
        self.assertListEqual(exp.tags(), ["other-tags"])
        self.assertDictEqual(exp.training_data().data(), self.fake_search_reply)

    # @patch('fedbiomed.researcher.federated_workflows._federated_workflow.Sec')
    def test_federated_workflow_09_secagg_setup(self):
        """Test secagg setup functionality and side effects"""

        with (
            patch(
                "fedbiomed.researcher.federated_workflows._federated_workflow.SecureAggregation",
                spec=SecureAggregation,
            ) as secure_aggregation_mock,
        ):
            FakeSecAgg.arg_train_arguments = {"secagg": "arguments"}
            _secagg = FakeSecAgg()

            # normal call
            _secagg.active = True
            secure_aggregation_mock.return_value = _secagg
            exp = FederatedWorkflow(secagg=True)

            secagg_args = exp.secagg_setup(["sampled-node-1", "sampled-node-2"])

            _secagg.setup.assert_called_once_with(
                parties=["sampled-node-1", "sampled-node-2"],
                experiment_id=exp.id,
                researcher_id=exp.researcher_id,
                insecure_validation=True,
            )
            self.assertDictEqual(secagg_args, {"secagg": "arguments"})

            # call with empty nodes list ...
            _secagg.reset_mock()
            exp = FederatedWorkflow(secagg=False)
            _secagg.active = False

            secagg_args = exp.secagg_setup([])

            _secagg.setup.assert_not_called()
            self.assertDictEqual(secagg_args, {})

            ## ... case where secagg is active but no nodes
            _secagg.reset_mock()
            exp = FederatedWorkflow(secagg=True)
            _secagg.active = True

            secagg_args = exp.secagg_setup([])

            _secagg.setup.assert_called_once_with(
                parties=[],
                experiment_id=exp.id,
                researcher_id=exp.researcher_id,
                insecure_validation=True,
            )

            self.assertDictEqual(secagg_args, {"secagg": "arguments"})

            # deactivate secagg, but one ndoe
            _secagg.reset_mock()
            _secagg.active = False

            exp = FederatedWorkflow(secagg=True)

            secagg_args = exp.secagg_setup(["sampled-nodes"])
            self.assertEqual(_secagg.setup.call_count, 0)
            self.assertDictEqual(secagg_args, {})

            # case where `_secagg.setup()` fails
            _secagg.reset_mock()
            _secagg.active = True
            _secagg.setup.return_value = False
            for nodes in [
                [],
                ["sampled-nodes"],
                ["sampled-node-1", "sampled-node-2"],
            ]:
                with self.assertRaises(FedbiomedSecureAggregationError):
                    exp.secagg_setup(nodes)

        # do not mock whole secagg module, and test that calling setup_secagg when secagg is inactive is a
        # noop that returns an empty dict
        with patch(
            "fedbiomed.researcher.federated_workflows._federated_workflow.SecureAggregation",
            spec=SecureAggregation,
        ) as secure_aggregation_mock:
            _secagg = FakeSecAgg()
            secure_aggregation_mock.return_value = _secagg
            exp = FederatedWorkflow(secagg=False)
            _secagg.active = False
            secagg_args = exp.secagg_setup([])
            self.assertEqual(_secagg.setup.call_count, 0)
            self.assertDictEqual(secagg_args, {})

    @patch("fedbiomed.researcher.federated_workflows._federated_workflow.open")
    @patch("fedbiomed.researcher.federated_workflows._federated_workflow.json.dump")
    @patch(
        "fedbiomed.researcher.federated_workflows._federated_workflow.choose_bkpt_file",
        return_value=("/bkpt-path", "bkpt-folder"),
    )
    def test_federated_workflow_10_breakpoint(
        self, mock_bkpt_file, mock_json_dump, mock_open
    ):
        # define attributes that will be saved in breakpoint
        _training_data = MagicMock(spec=fedbiomed.researcher.datasets.FederatedDataSet)
        _training_data.data.return_value = {"training": "data"}
        exp = FederatedWorkflow(
            training_data=_training_data,
        )
        exp.breakpoint(state={}, bkpt_number=1)
        # This also validates the breakpoint scheme: if this fails, please
        # consider updating the breakpoints version
        mock_json_dump.assert_called_once_with(
            {
                "id": exp.id,
                "breakpoint_version": str(__breakpoints_version__),
                "training_data": {"training": "data"},
                "experimentation_folder": exp.experimentation_folder(),
                "tags": exp.tags(),
                "nodes": exp.nodes(),
                "secagg": exp.secagg.save_state_breakpoint(),
                "node_state": exp._node_state_agent.save_state_breakpoint(),
            },
            mock_open.return_value.__enter__.return_value,
        )

        mock_open.side_effect = OSError

        with self.assertRaises(SystemExit):
            exp.breakpoint(state={}, bkpt_number=1)

    @patch("fedbiomed.researcher.federated_workflows._federated_workflow.open")
    @patch("fedbiomed.researcher.federated_workflows._federated_workflow.json.load")
    @patch(
        "fedbiomed.researcher.federated_workflows._federated_workflow.find_breakpoint_path",
        return_value=("/bkpt-path", "bkpt-folder"),
    )
    @patch(
        "fedbiomed.researcher.federated_workflows._federated_workflow.JoyeLibertSecureAggregation.load_state_breakpoint"
    )
    @patch(
        "fedbiomed.researcher.federated_workflows._federated_workflow.NodeStateAgent.load_state_breakpoint"
    )
    def test_federated_workflow_05_load_breakpoint(
        self,
        mock_node_state_load,
        mock_secagg_load,
        mock_bkpt_file,
        mock_json_load,
        mock_open,
    ):
        # Invalid argument should be string or None
        with self.assertRaises(SystemExit):
            exp, _ = FederatedWorkflow.load_breakpoint(breakpoint_folder_path=15)

        # Normal test case
        mock_secagg_load.return_value = MagicMock(spec=SecureAggregation)
        mock_node_state_load.return_value = MagicMock(
            spec=fedbiomed.researcher.node_state_agent.NodeStateAgent
        )
        mock_json_load.return_value = {
            "id": "exp-id",
            "breakpoint_version": str(__breakpoints_version__),
            "training_data": {"node1": [{"training": "data", "tags": "some-tags"}]},
            "experimentation_folder": "some-folder",
            "tags": ["some-tags"],
            "nodes": ["node1"],
            "secagg": {
                "class": "SecureAggregation",
                "module": "fedbiomed.researcher.secagg._secure_aggregation",
                "arguments": {"scheme": 2},
                "attributes": {},
                "attributes_states": {
                    "_SecureAggregation__secagg": {
                        "class": "LomSecureAggregation",
                        "module": "fedbiomed.researcher.secagg._secure_aggregation",
                        "arguments": {"active": False, "clipping_range": None},
                        "attributes": {
                            "_experiment_id": None,
                            "_parties": None,
                            "_dh": None,
                        },
                    }
                },
            },
            "node_state": {"node_state": "bkpt"},
            "downstream": "bkpt",
        }

        exp, saved_state = FederatedWorkflow.load_breakpoint()

        self.assertEqual(exp.id, "exp-id")
        self.assertEqual(
            exp.training_data().data(),
            {"node1": {"training": "data", "tags": "some-tags"}},
        )
        self.assertListEqual(exp.nodes(), ["node1"])
        self.assertListEqual(exp.tags(), ["some-tags"])
        self.assertEqual(saved_state["id"], "exp-id")
        self.assertEqual(
            saved_state["training_data"],
            {"node1": {"training": "data", "tags": "some-tags"}},
        )
        self.assertListEqual(saved_state["nodes"], ["node1"])
        self.assertListEqual(saved_state["tags"], ["some-tags"])
        self.assertEqual(saved_state["secagg"]["class"], "SecureAggregation")
        self.assertDictEqual(saved_state["node_state"], {"node_state": "bkpt"})
        self.assertEqual(exp.secagg.scheme, SecureAggregationSchemes.LOM)
        self.assertEqual(saved_state["downstream"], "bkpt")

        # Test error cases

        # If open raises an exception
        mock_open.side_effect = OSError
        with self.assertRaises(SystemExit):
            exp, _ = FederatedWorkflow.load_breakpoint()
        mock_open.side_effect = None

        # If saved state is not dict

        mock_json_load.return_value = ["list"]
        with self.assertRaises(SystemExit):
            exp, _ = FederatedWorkflow.load_breakpoint()

    def test_federated_workflow_06_all_federation_nodes(self):
        """Tests retrieving nodes"""

        ff = FederatedWorkflow()
        ff._fds = FederatedDataSet({"node-1": {}, "node-2": {}})
        nodes = ff.all_federation_nodes()
        self.assertListEqual(["node-1", "node-2"], nodes)

    def test_federated_workflow_07_filtered_federation_nodes(self):
        """Tests filereted federation nodes method"""

        ff = FederatedWorkflow()
        ff._nodes_filter = ["node-1"]
        ff._fds = FederatedDataSet({"node-1": {}, "node-2": {}})

        # Check filtered nodes
        nodes = ff.filtered_federation_nodes()
        self.assertListEqual(nodes, ["node-1"])

        # Check when there is no filter
        ff._nodes_filter = None
        nodes = ff.all_federation_nodes()
        self.assertListEqual(["node-1", "node-2"], nodes)

    def test_federated_workflow_08_experimentation_path(self):
        """Test retrieving experimentation path"""

        ff = FederatedWorkflow()
        path = ff.experimentation_path()
        self.assertTrue("experiment" in path)

    def test_federated_workflow_09_create_default_info_structure(self):
        """Tests creating default info structure"""
        ff = FederatedWorkflow()
        structure = ff._create_default_info_structure()
        self.assertDictEqual(structure, {"Arguments": [], "Values": []})

    def test_federated_workflow_10_info(self):
        """Tests printing experiment info"""
        ff = FederatedWorkflow()
        info, missing = ff.info()

        self.assertListEqual(
            info["Arguments"],
            [
                "Tags",
                "Nodes filter",
                "Training Data",
                "Experiment folder",
                "Experiment Path",
                "Secure Aggregation",
            ],
        )

        ff.info(missing="test")

    def test_federated_workflow_11_check_missing_objects(self):
        """Test missing objects"""

        fw = FederatedWorkflow()
        fw._training_ags = None
        missing = fw._check_missing_objects({"Training Args": None})
        self.assertEqual(missing, "- Training Data\n- Training Args\n")

    def test_federated_workflow_12_set_save_breakpoints(self):
        """Tests activating breakpoint"""

        fw = FederatedWorkflow()
        fw.set_save_breakpoints(True)

        # Invalid argument
        with self.assertRaises(SystemExit):
            fw.set_save_breakpoints("invalid")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
