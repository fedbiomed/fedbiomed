import argparse
import configparser
import io
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import fedbiomed
import fedbiomed.node.cli_utils
from fedbiomed.common.dataloadingplan import MapperBlock
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.node.cli import (
    DatasetArgumentParser,
    NodeCLI,
    NodeControl,
    TrainingPlanArgumentParser,
    start_node,
)
from fedbiomed.node.cli_utils import add_database
from fedbiomed.node.cli_utils._medical_folder_dataset import (
    add_medical_folder_dataset_from_cli,
    get_map_modalities2folders_from_cli,
)
from fedbiomed.node.config import NodeConfig

# from test_medical_datasets import patch_modality_glob, patch_is_modality_dir


class TestTrainingPlanArgumentParser(unittest.TestCase):
    """Test case for node cli dataset argument parse"""

    def setUp(self):
        self.parser = argparse.ArgumentParser()
        self.subparsers = self.parser.add_subparsers()
        self.tp_arg_pars = TrainingPlanArgumentParser(self.subparsers)
        self.tp_arg_pars._node = MagicMock()

    def test_01_training_plan_argument_parser_initialize(self):
        """Tests training plan parser initialization"""
        self.tp_arg_pars.initialize()
        self.assertTrue("training-plan" in self.subparsers.choices)

        tp_choices = (
            self.subparsers.choices["training-plan"]
            ._subparsers._group_actions[0]
            .choices
        )

        self.assertTrue("approve" in tp_choices)  # noqa
        self.assertTrue("list" in tp_choices)  # noqa
        self.assertTrue("delete" in tp_choices)  # noqa
        self.assertTrue("reject" in tp_choices)  # noqa
        self.assertTrue("view" in tp_choices)  # noqa
        self.assertTrue("update" in tp_choices)  # noqa

    def test_02_training_plan_argument_parser_execute(self):
        """Tests training plan argument parser actions"""

        self.tp_arg_pars.initialize()

        args = self.parser.parse_args(["training-plan", "delete"])
        self.tp_arg_pars.delete(args)
        self.tp_arg_pars.list()
        self.tp_arg_pars.update()

        with patch.object(fedbiomed.node.cli, "view_training_plan") as m:
            self.tp_arg_pars.view()
            m.assert_called_once()

        with patch.object(fedbiomed.node.cli, "register_training_plan") as m:
            self.tp_arg_pars.register()
            m.assert_called_once()

        with patch.object(fedbiomed.node.cli, "reject_training_plan") as m:
            args = self.parser.parse_args(["training-plan", "reject"])
            self.tp_arg_pars.reject(args)
            m.assert_called_once()


class TestDatasetArgumentParser(unittest.TestCase):
    """Test case for node cli dataset argument parse"""

    def setUp(self):
        self.parser = argparse.ArgumentParser()
        self.subparsers = self.parser.add_subparsers()
        self.dataset_arg_pars = DatasetArgumentParser(self.subparsers)
        self.node = MagicMock()
        self.dataset_arg_pars._node = self.node

    def test_01_dataset_argument_parser_initialize(self):
        """Test initialization"""

        self.dataset_arg_pars.initialize()
        self.assertTrue("dataset" in self.subparsers.choices)
        dataset_choices = (
            self.subparsers.choices["dataset"]._subparsers._group_actions[0].choices
        )

        self.assertTrue("add" in dataset_choices)  # noqa
        self.assertTrue("list" in dataset_choices)  # noqa
        self.assertTrue("delete" in dataset_choices)  # noqa

        self.assertTrue("--mnist" in dataset_choices["add"]._option_string_actions)
        self.assertTrue("--file" in dataset_choices["add"]._option_string_actions)

        self.assertTrue("--all" in dataset_choices["delete"]._option_string_actions)
        self.assertTrue("--mnist" in dataset_choices["delete"]._option_string_actions)

    def test_02_dataset_argument_parser_add(self):
        self.dataset_arg_pars.initialize()
        args = self.parser.parse_args(["dataset", "add"])

        with patch.object(fedbiomed.node.cli, "add_database") as m:
            self.dataset_arg_pars.add(args)
            m.assert_called_once()
            m.reset_mock()

            args = self.parser.parse_args(["dataset", "add", "--mnist", "test"])
            self.dataset_arg_pars.add(args)
            m.assert_called_once_with(
                self.node.dataset_manager, interactive=False, path="test"
            )
            m.reset_mock()

    def test_03_dataset_argument_parser_delete(self):
        self.dataset_arg_pars.initialize()

        with patch.object(fedbiomed.node.cli, "delete_database") as m:
            args = self.parser.parse_args(["dataset", "delete"])
            self.dataset_arg_pars.delete(args)
            m.assert_called_once_with(self.node.dataset_manager)

        with patch.object(fedbiomed.node.cli, "delete_all_database") as m:
            args = self.parser.parse_args(["dataset", "delete", "--all"])
            self.dataset_arg_pars.delete(args)
            m.assert_called_once_with(self.node.dataset_manager)
            m.reset_mock()

        with patch.object(fedbiomed.node.cli, "delete_database") as m:
            args = self.parser.parse_args(["dataset", "delete", "--mnist"])
            self.dataset_arg_pars.delete(args)
            m.assert_called_once_with(self.node.dataset_manager, interactive=False)

    def test_04_dataset_argument_parser_list(self):
        self.dataset_arg_pars.initialize()
        args = self.parser.parse_args(["dataset", "list"])
        self.dataset_arg_pars.list(args)

        self.node.dataset_manager.list_my_data.return_value = []
        self.node.dataset_manager.list_my_data.assert_called_once_with(verbose=True)


class TestNodeControl(unittest.TestCase):
    """Test case for node control parser"""

    def setUp(self):
        self.parser = argparse.ArgumentParser()
        self.subparsers = self.parser.add_subparsers()
        self.control = NodeControl(self.subparsers)
        self.node = MagicMock()
        self.control._node = self.node

    def test_01_node_control_initialize(self):
        """Tests initialize"""

        self.control.initialize()
        self.assertTrue("start" in self.subparsers.choices)
        self.assertTrue(
            "--gpu-only" in self.subparsers.choices["start"]._option_string_actions
        )  # noqa
        self.assertTrue(
            "--gpu-num" in self.subparsers.choices["start"]._option_string_actions
        )  # noqa
        self.assertTrue(
            "--gpu" in self.subparsers.choices["start"]._option_string_actions
        )  # noqa

    @patch("fedbiomed.node.cli.Process")
    def test_02_node_control_start(self, process):
        self.control.initialize()
        args = self.parser.parse_args(["start"])
        os.environ["FEDBIOMED_ACTIVE_NODE_ID"] = "test-node-id"

        self.control.start(args)
        process.assert_called_once()

        process.return_value.join.side_effect = KeyboardInterrupt
        process.return_value.is_alive.side_effect = [True, False]
        with self.assertRaises(SystemExit):
            self.control.start(args)

    @patch("fedbiomed.node.cli.Node", autospec=True)
    def test_03_node_control__start(self, mock_node):
        """Tests node start"""

        cfg = configparser.ConfigParser()
        cfg["security"] = {
            "training_plan_apprival": "true",
            "allow_default_training_plan": "true",
        }
        cfg["default"] = {"id": "test-id"}

        mock_node.return_value.tp_security_manager = MagicMock()

        with tempfile.TemporaryDirectory() as temp_:
            config = NodeConfig(temp_)
            config._cfg = cfg
            args = {"gpu": False}
            config._cfg["security"]["training_plan_approval"] = "false"
            start_node("config.ini", args)
            mock_node.return_value.task_manager.assert_called_once()

            with patch.object(fedbiomed.node.cli, "logger") as logger:
                mock_node.return_value.task_manager.side_effect = FedbiomedError
                start_node("config.ini", args)
                logger.critical.assert_called_once()
                logger.critical.reset_mock()

                mock_node.return_value.task_manager.side_effect = Exception
                start_node("config.ini", args)
                logger.critical.assert_called_once()


class TestNodeCLI(unittest.TestCase):
    """Tests main NodeCLI"""

    @patch("builtins.input")
    def test_01_node_cli_init(self, input_patch):
        """Tests initialization"""
        input_patch.return_value = "y"
        # import sys
        # sys.argv.append('-y')
        # remove any `fbm-node` folder already existing
        shutil.rmtree("fbm-node", ignore_errors=True)
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.node_cli = NodeCLI()
            self.node_cli.parse_args(
                ["--path", os.path.join(str(tmpdirname), "fbm-node"), "dataset", "list"]
            )
        # sys.argv.remove('-y')


class TestCli(unittest.TestCase):
    @staticmethod
    def mock_cli_input(x):
        """Requires that each test defines TestCli.inputs as a list"""
        return TestCli.inputs.pop(0)

    @staticmethod
    def patch_modality_glob(x, y):
        # We are globbing all subject folders
        for f in (
            [Path("T1philips"), Path("T2"), Path("label")]
            + [Path("T1siemens"), Path("T2"), Path("label")]
            + [Path("non-existing-modality")]
        ):
            yield f

    def setUp(self) -> None:
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

    def tearDown(self) -> None:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    @unittest.skip("Deprecated")
    def test_cli_01_add_database_medical_folder(self):
        database_inputs = ["test-db-name", "test-tag1,test-tag2", "", "test-dlp-name"]
        medical_folder_inputs = [
            "y",
            "0",
            "y",
            "1",
            "1",
            "2",
            "3",
            "0",
            "Tnon-exist",
            "y",
            "4",
        ]

        with (
            patch("pathlib.Path.glob", new=TestCli.patch_modality_glob) as patched_glob,
            patch("pathlib.Path.is_dir", return_value=True) as patched_dir,
            patch(
                "fedbiomed.common.dataset.MedicalFolderBase.demographics_column_names",
                return_value=["col1", "col2"],
            ) as patched_column_names,
            patch(
                "fedbiomed.common.dataset.MedicalFolderBase.validate_MedicalFolder_root_folder",
                return_value=Path("some/valid/path"),
            ) as patched_validate_root,
            patch(
                "fedbiomed.node.cli_utils._medical_folder_dataset.validated_path_input",
                return_value="some/valid/path",
            ) as patched_val_path_in,
            patch(
                "fedbiomed.node.cli_utils._io.input", return_value="5"
            ) as patched_cli_input,
            patch(
                "fedbiomed.node.cli_utils._database.input",
                new=lambda x: database_inputs.pop(0),
            ) as patched_db_input,
            patch(
                "fedbiomed.node.cli_utils._medical_folder_dataset.input",
                new=lambda x: medical_folder_inputs.pop(0),
            ) as patched_medical_folder_input,
        ):
            # Need to override equality test to enable assert_called_once_with to function properly
            def test_mapper_eq(self, other):
                return self.map == other.map

            MapperBlock.__eq__ = test_mapper_eq

            fedbiomed.node.cli_utils.add_database = MagicMock()
            add_database(MagicMock())

            dlb = MapperBlock()
            dlb.map = {
                "T1": ["T1philips", "T1siemens"],
                "T2": ["T2"],
                "label": ["label"],
                "Tnon-exist": ["non-existing-modality"],
            }


class TestMedicalFolderCliUtils(unittest.TestCase):
    @staticmethod
    def mock_input(x):
        return TestMedicalFolderCliUtils.inputs.pop(0)

    @staticmethod
    def mock_validated_input(type):
        return "some/valid/path"

    def setUp(self) -> None:
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

    def tearDown(self) -> None:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    @patch(
        "fedbiomed.node.cli_utils._medical_folder_dataset.input",
        new=mock_input.__func__,
    )
    @unittest.skip("Deprecated")
    def test_medical_folder_cli_utils_01_get_map_modalities2folders_from_cli(self):
        modality_folder_names = ["Should map to T1"]
        # scenario 1: 'Should map to T1' <-> 'T1'. Assumes T1 is second in the list of modalities provided by default
        TestMedicalFolderCliUtils.inputs = ["1"]
        dlb = get_map_modalities2folders_from_cli(modality_folder_names)
        self.assertDictEqual(dlb.map, {"T1": ["Should map to T1"]})

        # scenario 2: wrong inputs first, then same as above
        TestMedicalFolderCliUtils.inputs = ["wrong", "5", "1"]
        dlb = get_map_modalities2folders_from_cli(modality_folder_names)
        self.assertDictEqual(dlb.map, {"T1": ["Should map to T1"]})

        # scenario 3: add new name
        modality_folder_names = ["Should map to T1", "Should map to Tnew"]
        TestMedicalFolderCliUtils.inputs = ["1", "0", "Tnew", "y", "4"]
        dlb = get_map_modalities2folders_from_cli(modality_folder_names)
        self.assertDictEqual(
            dlb.map, {"T1": ["Should map to T1"], "Tnew": ["Should map to Tnew"]}
        )

        # scenario 4: More complexity with some mistakes added in
        modality_folder_names = [
            "Should map to T1",
            "Should map to Tnew",
            "Should also map to T1",
            "Should map to T2",
        ]
        TestMedicalFolderCliUtils.inputs = [
            "1",
            "0",
            "Tmistake",
            "n",
            "Tnew",
            "",
            "4",
            "5",
            "1",
            "2",
        ]
        dlb = get_map_modalities2folders_from_cli(modality_folder_names)
        self.assertDictEqual(
            dlb.map,
            {
                "T1": ["Should map to T1", "Should also map to T1"],
                "Tnew": ["Should map to Tnew"],
                "T2": ["Should map to T2"],
            },
        )

    @patch(
        "fedbiomed.node.cli_utils._medical_folder_dataset.input",
        new=mock_input.__func__,
    )
    @patch(
        "fedbiomed.node.cli_utils._medical_folder_dataset.validated_path_input",
        new=mock_validated_input.__func__,
    )
    @patch(
        "fedbiomed.common.dataset.MedicalFolderBase.validate_MedicalFolder_root_folder",
        return_value=Path("some/valid/path"),
    )
    @patch(
        "fedbiomed.common.dataset.MedicalFolderBase.demographics_column_names",
        return_value=["col1", "col2"],
    )
    # @patch("pathlib.Path.glob", new=patch_modality_glob)
    # @patch("pathlib.Path.is_dir", new=patch_is_modality_dir)
    @unittest.skip("Deprecated")
    def test_medical_folder_cli_utils_02_load_medical_folder_dataset_from_cli(
        self, patch_validated_path_input, patch_validate_root_folder
    ):
        # Scenario 1:
        #    - no pre-existing dataset parameters or data loading plan
        #    - user selects a demographics file
        #    - user wants to configure a data loading plan
        TestMedicalFolderCliUtils.inputs = [
            "y",
            "0",
            "y",
            "1",
            "1",
            "2",
            "3",
            "0",
            "Tnon-exist",
            "y",
            "4",
        ]

        path, dataset_parameters, dlp = add_medical_folder_dataset_from_cli(
            True, None, None
        )
        self.assertEqual(path, "some/valid/path")
        self.assertDictEqual(
            dataset_parameters, {"tabular_file": "some/valid/path", "index_col": 0}
        )
        # TODO : test with DLP, when CLI DLP supported by future version
        # self.assertIn(MedicalFolderLoadingBlockTypes.MODALITIES_TO_FOLDERS, dlp)
        # self.assertDictEqual(dlp[MedicalFolderLoadingBlockTypes.MODALITIES_TO_FOLDERS].map, {
        #    'T1': ['T1philips', 'T1siemens'],
        #    'T2': ['T2'],
        #    'Tnon-exist': ['non-existing-modality'],
        #    'label': ['label']
        # })

        # Scenario 2:
        #    - no pre-existing dataset parameters or data loading plan
        #    - user selects a demographics file
        #    - user does not configure a data loading plan
        TestMedicalFolderCliUtils.inputs = ["y", "0", ""]

        path, dataset_parameters, dlp = add_medical_folder_dataset_from_cli(
            True, None, None
        )
        self.assertEqual(path, "some/valid/path")
        self.assertDictEqual(
            dataset_parameters, {"tabular_file": "some/valid/path", "index_col": 0}
        )
        self.assertIsNone(dlp)

        # Scenario 2:
        #    - no pre-existing dataset parameters or data loading plan
        #    - user does not select a demographics file
        #    - user does not configure a data loading plan
        TestMedicalFolderCliUtils.inputs = ["n", ""]

        path, dataset_parameters, dlp = add_medical_folder_dataset_from_cli(
            True, None, None
        )
        self.assertEqual(path, "some/valid/path")
        self.assertDictEqual(dataset_parameters, {})
        self.assertIsNone(dlp)


if __name__ == "__main__":
    unittest.main()
