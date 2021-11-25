import unittest
from unittest.mock import patch, MagicMock, mock_open, Mock, PropertyMock
import os
import tempfile
import shutil
import json
from typing import Union

# import a fake environment for tests bafore importing other files
import testsupport.mock_common_environ

from fedbiomed.researcher.environ import environ
from fedbiomed.researcher.experiment import Experiment


def create_file(file_name: str):
    """creates a file in the specified path `file_name`

    Args:
        file_name (str): path of the file
    """
    with open(file_name, "w") as f:
        f.write("this is a test- file. \
            This file should be removed at the end of unit tests")

def load_json(file: str) -> Union[None, Exception]:
    """tests if a JSON file is parsable

    Args:
        file (str): path name of the json file to load
    Returns:

    """
    try:
        with open(file, "r") as f:
            json.load(f)
        return None
    except Exception as err:
        return err


# FIXME: it seems it is more an integration test than a unit test.
# an improvement can be to 'patch' Job class instead of calling it.
class TestStateExp(unittest.TestCase):

    def setUp(self):
        fds = MagicMock()
        fds.keys = MagicMock(return_value={})

        try:
            shutil.rmtree(os.path.join(environ['VAR_DIR'], "breakpoints"))
            # clean up existing breakpoints
        except FileNotFoundError:
            pass
        self.patcher = patch('fedbiomed.researcher.requests.Requests.__init__',
                             return_value=None)
        self.patcher2 = patch('fedbiomed.researcher.requests.Requests.search',
                              return_value=fds)
        self.patcher3 = patch('fedbiomed.common.repository.Repository.upload_file',
                              return_value={"file": environ['UPLOADS_URL']})
        self.patcher_monitor = patch('fedbiomed.researcher.experiment.Monitor',
                                     return_value=None)

        self.patcher.start()
        self.patcher2.start()
        self.patcher3.start()
        self.patcher_monitor.start()
        model_file = MagicMock(return_value=None)

        model_file.save_code = MagicMock(return_value=None)
        self.test_exp = Experiment(
            'some_tags', model_class=model_file, save_breakpoints=True
        )

        self.test_exp._create_breakpoint_exp_folder()

        self.test_exp._state_root_folder = environ['VAR_DIR']  # changing value of
        # the root folder

    def tearDown(self) -> None:

        self.patcher.stop()
        self.patcher2.stop()
        self.patcher3.stop()
        self.patcher_monitor.stop()
        try:
            shutil.rmtree(os.path.join(environ['VAR_DIR'], "breakpoints"))
            # (above) remove files created during these unit tests
        except FileNotFoundError:
            pass


    def test_save_states(self):
        """tests `save_states` private method:
        1. if model file is copied from temporary folder to breakpoint folder
        2. if state file created is json loadable
        """

        # Lets mock `node_selection_strategy`
        self.test_exp._node_selection_strategy = MagicMock(return_value=None)
        self.test_exp._node_selection_strategy.save_state = MagicMock(return_value={})
        with tempfile.TemporaryDirectory(dir=environ['TMP_DIR']) as tmpdirname:
            def return_state(x=0):
                '''mimicking job.py 'save_state' method'''
                pass
            tempfile_path = os.path.join(tmpdirname, 'test_save')
            create_file(tempfile_path)
            self.test_exp._job.save_state = return_state
            self.test_exp._job.state = {"model_path": str(tempfile_path),
                                        'params_path': {}}

            self.test_exp._save_state()
            exp_path = os.path.join(environ['VAR_DIR'],
                                    "breakpoints",
                                    "Experiment_0",
                                    'breakpoint_0')
            self.assertTrue(os.path.isdir(exp_path))  # test if folder used
            # for saving the breakpoint exists
            test_model_path = os.path.join(exp_path, "breakpoint_0.json")

        # test if file containing state of experiment in breakpoint folder
        # is JSON loadable
        val = load_json(test_model_path)
        self.assertIs(val, None)

    def test_create_breakpoint(self,
                               breakpoint_folder_name: str="breakpoint_"):
        """
        Tests method `_create_breakpoint_file_and_folder`. Checks the correct
        spelling of breakpoint and state file.

        Args:
            breakpoint_folder_name (str, optional): [description]. Defaults to "breakpoint_".
        """

        bkpt_folder, bkpt_file = self.test_exp._create_breakpoint_file_and_folder(
                                                                    round=0)

        self.assertEqual(os.path.basename(bkpt_folder),
                         breakpoint_folder_name + str(0))
        self.assertEqual(bkpt_file,
                         breakpoint_folder_name + str(0) + ".json")

        bkpt_folder, bkpt_file = self.test_exp._create_breakpoint_file_and_folder(
                                                                    round=2)
        self.assertEqual(os.path.basename(bkpt_folder),
                         breakpoint_folder_name + str(2))
        self.assertEqual(bkpt_file,
                         breakpoint_folder_name + str(2) + ".json")

    def test_private_get_latest_file(self):
        """tests if `_get_latest_file` returns more recent
        file"""

        # test 1
        files = ["Experiment_0",
                 "Experiment_4",
                 "EXperiment_5",
                 "blabla",
                 "99_blabla"]

        pathfile_test = "/path/to/a/file"

        latest_file = Experiment._get_latest_file(pathfile_test,
                                                  files,
                                                  only_folder=False)
        self.assertEqual(files[2], latest_file)

        # test 2: in this test, we patch isir builtin function
        # so it returns always `True` when called
        patcher_builtin_os_path_isdir = patch("os.path.isdir",
                                              return_value=True)
        patcher_builtin_os_path_isdir.start()
        latest_file = Experiment._get_latest_file(pathfile_test,
                                                  files,
                                                  only_folder=True)
        self.assertEqual(files[2], latest_file)
        patcher_builtin_os_path_isdir.stop()
        # test 3
        files = []
        latest_file = Experiment._get_latest_file(pathfile_test,
                                                  files,
                                                  only_folder=False)
        self.assertEqual(latest_file, None)

        # test 4: test if exception is raised
        files = ['q', 'foo', 'bar']

        self.assertRaises(FileNotFoundError,
                          Experiment._get_latest_file,
                          pathfile_test,
                          files,
                          only_folder=False)

    def test_private_import_module(self):
        args = {"class": "myclass",
                "module": "module.containing.my.class",
                "parameters": None}

        # test 1: default
        import_str = Experiment._import_module(args)
        self.assertEqual(import_str,
                         'from module.containing.my.class import myclass')
        # test 2: custom module
        args = {"class": "myclass",
                "module": "custom",
                "parameters": None}
        import_str = Experiment._import_module(args)
        self.assertEqual(import_str, 'import myclass')

    @patch('fedbiomed.researcher.job.Job._load_training_replies')
    def test_private_load_training_replies(self,
                                           path_job_load_training_replies):
        path_job_load_training_replies.return_value = None
        tr_replies = {1: [{"foo": "bar"}]}
        pr_path = ["/path/to/file", "/path/to/file"]
        self.test_exp._load_training_replies(tr_replies, pr_path)
        # test if Job's `_load_training_replies` has been called once
        path_job_load_training_replies.assert_called_once()

    @patch('fedbiomed.researcher.job.Job._load_training_replies')
    @patch('fedbiomed.researcher.job.Job.__init__')
    @patch('fedbiomed.researcher.experiment.eval')
    @patch('fedbiomed.researcher.experiment.Experiment._import_module')
    @patch('json.load')
    @patch("builtins.open")
    @patch('fedbiomed.researcher.experiment.Experiment._find_breakpoint_path')
    def test_load_breakpoint(self,
                             patch_find_breakpoint_path,
                             patch_builtin_open,
                             patch_json_load,
                             patch_import_module,
                             patch_builtin_eval,
                             patch_job_init,
                             patch_job_load_training_replies
                             ):

        values = ["/path/to/breakpoint/folder", "my_breakpoint.json"]
        dummy_agg = {"class":None,
                     "Module":None}
        loaded_states = {
            "node_selection_strategy": dummy_agg,
            "aggregator": dummy_agg,
            "tags": ["some_tags"],
            "node_id": "m_node_id",
            "model_class": "my_model_class",
            "model_path": "/path/to/model/file",
            "model_args": {},
            "training_args":{},
            "round_number": 1,
            "round_number_due":3,
            "training_data": {},
            "job_id": "1234",
            "researcher_id": '1234',
            "params_path": [],
            "training_replies": {"1":[{}, {}]}

        }
        patch_find_breakpoint_path.return_value = values
        patch_builtin_open.return_value = MagicMock()
        patch_json_load.return_value = loaded_states
        patch_import_module.return_value = "import abc"  # not sure it is a good idea
        patch_builtin_eval.return_value = MagicMock()
        patch_job_init.return_value = None
        patch_job_load_training_replies.return_value = MagicMock()

        bkpt_folder = "/path/to/breakpoint/folder"
        loaded_exp = Experiment.load_breakpoint(bkpt_folder)
        print(type(loaded_exp))

        # tests
        patch_json_load.assert_called_once()  # check if patched
        # json has been called
        self.assertTrue(isinstance(loaded_exp, Experiment))
        self.assertEqual(loaded_exp._round_init,
                         loaded_states.get('round_number'))
        self.assertEqual(loaded_exp._job._id,
                         loaded_states.get('job_id'))
        self.assertEqual(loaded_exp._rounds,
                         loaded_states.get('round_number_due'))

    @patch('os.path.isdir')
    @patch('os.listdir')
    def test_private_find_breakpoint_path_1(self,
                                            patch_os_listdir,
                                            patch_os_path_isdir
                                            ):
        # test 1 : test if results are corrects  if path
        # to breakpoint has been given by user
        bkpt_folder = "/path/to/breakpoint"
        patch_os_listdir.return_value = ['breakpoint_1234.json',
                                         "another_file"]
        patch_os_path_isdir.return_value = True

        bkpt_folder_out, state_file = Experiment._find_breakpoint_path(bkpt_folder)
        self.assertEqual(bkpt_folder, bkpt_folder_out)
        self.assertEqual(state_file, 'breakpoint_1234.json')

    @patch('fedbiomed.researcher.experiment.Experiment._get_latest_file')
    @patch('os.path.isdir')
    @patch('os.listdir')
    def test_private_find_breakpoint_path_2(self,
                                            patch_os_listdir,
                                            patch_os_path_isdir,
                                            patch_get_latest_file
                                            ):
        # test 2 : test if path to breakpoint has not been given by user
        # ie set to None
        patch_os_listdir.return_value = ['breakpoint_1234.json',
                                         "another_file"]
        patch_os_path_isdir.return_value = True
        patch_get_latest_file.return_value = "breakpoint"
        latest_bkpt_folder = os.path.join(environ['BREAKPOINTS_DIR'],
                                          'breakpoint',
                                          'breakpoint')

        bkpt_folder_out, state_file = Experiment._find_breakpoint_path(None)
        self.assertEqual(state_file, 'breakpoint_1234.json')
        self.assertEqual(bkpt_folder_out, latest_bkpt_folder)

    @patch('os.path.isfile')
    @patch('os.path.isdir')
    @patch('os.listdir')
    def test_private_find_breakpoint_path_raise_err1(self,
                                                     patch_os_listdir,
                                                     patch_os_path_isdir,
                                                     patch_os_path_isfile):
        # triggers error: FileNotFoundError, error is not a folder
        # but a file
        bkpt_folder = "/path/to/breakpoint"
        patch_os_listdir.return_value = ['breakpoint_1234.json',
                                         "another_file"]
        patch_os_path_isdir.return_value = False
        patch_os_path_isfile.return_value = False
        self.assertRaises(FileNotFoundError,
                          Experiment._find_breakpoint_path,
                          bkpt_folder)
    @patch('os.path.isfile')
    @patch('os.path.isdir')
    @patch('os.listdir')
    def test_private_find_breakpoint_path_raise_err2(self,
                                                     patch_os_listdir,
                                                     patch_os_path_isdir,
                                                     patch_os_path_isfile):
        # triggers error: FileNotFoundError (folder not found)
        #
        bkpt_folder = "/path/to/breakpoint"
        patch_os_listdir.return_value = ['breakpoint_1234.json',
                                         "another_file"]
        patch_os_path_isdir.return_value = False
        patch_os_path_isfile.return_value = True
        self.assertRaises(FileNotFoundError,
                          Experiment._find_breakpoint_path,
                          bkpt_folder)

    @patch('fedbiomed.researcher.experiment.Experiment._get_latest_file')
    @patch('os.path.isdir')
    @patch('os.listdir')
    def test_private_find_breakpoint_raise_err_3(self,
                                                 patch_os_listdir,
                                                 patch_os_path_isdir,
                                                 patch_get_latest_file
                                                 ):
        # test 3 : test if rerror is raised when json file
        # not found in a breakpoint folder specified by user
        bkpt_folder = "/path/to/breakpoint"
        patch_os_listdir.return_value = ['one_file',
                                         "another_file"]
        patch_os_path_isdir.return_value = True
        patch_get_latest_file.return_value = "breakpoint"

        self.assertRaises(FileNotFoundError,
                          Experiment._find_breakpoint_path,
                          bkpt_folder)

    def test_private_find_breakpoint_raise_err_4(self):
        # test 4 : test if rerror is raised when latest
        # file has not been foud
        self.assertRaises(FileNotFoundError,
                          Experiment._find_breakpoint_path,
                          None)

if __name__ == '__main__':  # pragma: no cover
    unittest.main()
