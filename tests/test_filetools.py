# Managing NODE, RESEARCHER environ mock before running tests
from testsupport.delete_environ import delete_environ
# Delete environ. It is necessary to rebuild environ for required component
delete_environ()
# overload with fake environ for tests
import testsupport.mock_common_environ
# Import environ for researcher, since tests will be running for researcher component
from fedbiomed.researcher.environ import environ

import unittest
from unittest.mock import patch
import os
import shutil

from fedbiomed.researcher import filetools


class TestFiletools(unittest.TestCase):

    def setUp(self):

        self.testdir = environ['TMP_DIR']

        # delete and re-create empty the test directory
        try:
            shutil.rmtree(self.testdir)  
        except FileNotFoundError:
            pass
        os.makedirs(self.testdir) 

        self.patchers = [
            # empty now
        ]
        
        for patcher in self.patchers:
            patcher.start()

    def tearDown(self) -> None:

        for patcher in self.patchers:
            patcher.stop()

        try:
            shutil.rmtree(self.testdir)
        except FileNotFoundError:
            pass


    def test_choose_bkpt_file(self):
        """
        Tests method `choose_bkpt_file`.
        Checks the correct spelling of breakpoint and state file.
        """

        breakpoint_folder_name = "dummy_dir_"

        bkpt_folder, bkpt_file = filetools.choose_bkpt_file(self.testdir, round=0)

        self.assertEqual(os.path.basename(bkpt_folder),
                         breakpoint_folder_name + str(0))
        self.assertEqual(bkpt_file,
                         breakpoint_folder_name + str(0) + ".json")

        bkpt_folder, bkpt_file = filetools.choose_bkpt_file(self.testdir, round=2)

        self.assertEqual(os.path.basename(bkpt_folder),
                         breakpoint_folder_name + str(2))
        self.assertEqual(bkpt_file,
                         breakpoint_folder_name + str(2) + ".json")

#    def test_private_get_latest_file(self):
#        """tests if `_get_latest_file` returns more recent
#        file"""
#
#        # test 1
#        files = ["Experiment_0",
#                 "Experiment_4",
#                 "EXperiment_5",
#                 "blabla",
#                 "99_blabla"]
#
#        pathfile_test = "/path/to/a/file"
#
#        latest_file = Experiment._get_latest_file(pathfile_test,
#                                                  files,
#                                                  only_folder=False)
#        self.assertEqual(files[2], latest_file)
#
#        # test 2: in this test, we patch isir builtin function
#        # so it returns always `True` when called
#        patcher_builtin_os_path_isdir = patch("os.path.isdir",
#                                              return_value=True)
#        patcher_builtin_os_path_isdir.start()
#        latest_file = Experiment._get_latest_file(pathfile_test,
#                                                  files,
#                                                  only_folder=True)
#        self.assertEqual(files[2], latest_file)
#        patcher_builtin_os_path_isdir.stop()
#        # test 3
#        files = []
#        latest_file = Experiment._get_latest_file(pathfile_test,
#                                                  files,
#                                                  only_folder=False)
#        self.assertEqual(latest_file, None)
#
#        # test 4: test if exception is raised
#        files = ['q', 'foo', 'bar']
#
#        self.assertRaises(FileNotFoundError,
#                          Experiment._get_latest_file,
#                          pathfile_test,
#                          files,
#                          only_folder=False)
#
#    @patch('fedbiomed.researcher.job.Job._load_training_replies')
#    def test_private_load_training_replies(self,
#                                           path_job_load_training_replies):
#        path_job_load_training_replies.return_value = None
#        tr_replies = {1: [{"foo": "bar"}]}
#        pr_path = ["/path/to/file", "/path/to/file"]
#        self.test_exp._load_training_replies(tr_replies, pr_path)
#        # test if Job's `_load_training_replies` has been called once
#        path_job_load_training_replies.assert_called_once()
#
#    @patch('os.path.isdir')
#    @patch('os.listdir')
#    def test_private_find_breakpoint_path_1(self,
#                                            patch_os_listdir,
#                                            patch_os_path_isdir
#                                            ):
#        # test 1 : test if results are corrects  if path
#        # to breakpoint has been given by user
#        bkpt_folder = "/path/to/breakpoint"
#        patch_os_listdir.return_value = ['breakpoint_1234.json',
#                                         "another_file"]
#        patch_os_path_isdir.return_value = True
#
#        bkpt_folder_out, state_file = Experiment._find_breakpoint_path(bkpt_folder)
#        self.assertEqual(bkpt_folder, bkpt_folder_out)
#        self.assertEqual(state_file, 'breakpoint_1234.json')
#
#    @patch('fedbiomed.researcher.experiment.Experiment._get_latest_file')
#    @patch('os.path.isdir')
#    @patch('os.listdir')
#    def test_private_find_breakpoint_path_2(self,
#                                            patch_os_listdir,
#                                            patch_os_path_isdir,
#                                            patch_get_latest_file
#                                            ):
#        # test 2 : test if path to breakpoint has not been given by user
#        # ie set to None
#        patch_os_listdir.return_value = ['breakpoint_1234.json',
#                                         "another_file"]
#        patch_os_path_isdir.return_value = True
#        patch_get_latest_file.return_value = "breakpoint"
#        latest_bkpt_folder = os.path.join(environ['EXPERIMENTS_DIR'],
#                                          'breakpoint',
#                                          'breakpoint')
#
#        bkpt_folder_out, state_file = Experiment._find_breakpoint_path(None)
#        self.assertEqual(state_file, 'breakpoint_1234.json')
#        self.assertEqual(bkpt_folder_out, latest_bkpt_folder)
#
#    @patch('os.path.isfile')
#    @patch('os.path.isdir')
#    @patch('os.listdir')
#    def test_private_find_breakpoint_path_raise_err1(self,
#                                                     patch_os_listdir,
#                                                     patch_os_path_isdir,
#                                                     patch_os_path_isfile):
#        # triggers error: FileNotFoundError, error is not a folder
#        # but a file
#        bkpt_folder = "/path/to/breakpoint"
#        patch_os_listdir.return_value = ['breakpoint_1234.json',
#                                         "another_file"]
#        patch_os_path_isdir.return_value = False
#        patch_os_path_isfile.return_value = False
#        self.assertRaises(FileNotFoundError,
#                          Experiment._find_breakpoint_path,
#                          bkpt_folder)
#    @patch('os.path.isfile')
#    @patch('os.path.isdir')
#    @patch('os.listdir')
#    def test_private_find_breakpoint_path_raise_err2(self,
#                                                     patch_os_listdir,
#                                                     patch_os_path_isdir,
#                                                     patch_os_path_isfile):
#        # triggers error: FileNotFoundError (folder not found)
#        #
#        bkpt_folder = "/path/to/breakpoint"
#        patch_os_listdir.return_value = ['breakpoint_1234.json',
#                                         "another_file"]
#        patch_os_path_isdir.return_value = False
#        patch_os_path_isfile.return_value = True
#        self.assertRaises(FileNotFoundError,
#                          Experiment._find_breakpoint_path,
#                          bkpt_folder)
#
#    @patch('fedbiomed.researcher.experiment.Experiment._get_latest_file')
#    @patch('os.path.isdir')
#    @patch('os.listdir')
#    def test_private_find_breakpoint_raise_err_3(self,
#                                                 patch_os_listdir,
#                                                 patch_os_path_isdir,
#                                                 patch_get_latest_file
#                                                 ):
#        # test 3 : test if rerror is raised when json file
#        # not found in a breakpoint folder specified by user
#        bkpt_folder = "/path/to/breakpoint"
#        patch_os_listdir.return_value = ['one_file',
#                                         "another_file"]
#        patch_os_path_isdir.return_value = True
#        patch_get_latest_file.return_value = "breakpoint"
#
#        self.assertRaises(FileNotFoundError,
#                          Experiment._find_breakpoint_path,
#                          bkpt_folder)
#
#    def test_private_find_breakpoint_raise_err_4(self):
#        # test 4 : test if rerror is raised when latest
#        # file has not been foud
#        self.assertRaises(FileNotFoundError,
#                          Experiment._find_breakpoint_path,
#                          None)

if __name__ == '__main__':  # pragma: no cover
    unittest.main()
