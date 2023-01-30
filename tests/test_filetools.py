import unittest
from unittest.mock import patch
import os
import shutil

#############################################################
# Import ResearcherTestCase before importing any FedBioMed Module
from testsupport.base_case import ResearcherTestCase
#############################################################

from fedbiomed.researcher.environ import environ
from fedbiomed.researcher import filetools


class TestFiletools(ResearcherTestCase):

    def setUp(self):

        self.testdir = environ['EXPERIMENTS_DIR']

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

    def test_filetools_01_create_exp_folder(self):
        """
        Tests method `create_exp_folder`
        """

        # OK giving a folder name
        exp_folder = 'my_exp_folder'
        self.assertEqual(exp_folder, filetools.create_exp_folder(exp_folder))
        self.assertTrue(os.path.isdir(os.path.join(self.testdir, exp_folder)))

        # KO folder cannot be created
        os.chmod(self.testdir, 0o500)
        exp_folder = 'bad_luck_for_you'
        self.assertRaises(PermissionError, filetools.create_exp_folder, exp_folder)
        os.chmod(self.testdir, 0o700)

        # OK not choosing folder name, receiving default
        return_folder = filetools.create_exp_folder()
        self.assertEqual(return_folder, 'Experiment_0001')
        self.assertTrue(os.path.isdir(os.path.join(self.testdir, return_folder)))

        # KO giving a path instead of folder name
        exp_folders = [
            './toto/dir',
            'a/b',
            '/tmp/var/',
            self.testdir
        ]
        for testpath in exp_folders:
            self.assertRaises(ValueError, filetools.create_exp_folder, testpath)

        # KO cannot create EXPERIMENTS_DIR
        saved_expdir = environ['EXPERIMENTS_DIR']
        environ['EXPERIMENTS_DIR'] = os.path.join(self.testdir, 'subdir')
        os.chmod(self.testdir, 0o500)
        self.assertRaises(PermissionError, filetools.create_exp_folder, 'a_good_folder')
        os.chmod(self.testdir, 0o700)
        environ['EXPERIMENTS_DIR'] = saved_expdir

    def test_filetools_02_choose_bkpt_file(self):
        """
        Tests method `choose_bkpt_file`.
        Checks the correct spelling of breakpoint and state file.
        """

        breakpoint_folder_name = "breakpoint_"

        bkpt_folder, bkpt_file = filetools.choose_bkpt_file(self.testdir, round_=0)

        self.assertEqual(os.path.basename(bkpt_folder),
                         breakpoint_folder_name + str("{:04d}".format(0)))
        self.assertEqual(bkpt_file,
                         breakpoint_folder_name + str("{:04d}".format(0)) + ".json")
        self.assertTrue(os.path.isdir(bkpt_folder))

        bkpt_folder, bkpt_file = filetools.choose_bkpt_file(self.testdir, round_=2)

        self.assertEqual(os.path.basename(bkpt_folder),
                         breakpoint_folder_name + str("{:04d}".format(2)))
        self.assertEqual(bkpt_file,
                         breakpoint_folder_name + str("{:04d}".format(2)) + ".json")
        self.assertTrue(os.path.isdir(bkpt_folder))

    @patch('os.mkdir')
    def test_filetools_03_choose_bkpt_file_exception(self, mock_mkdir):
        """ Test PermissionError and OSError while choosing breakpoint file """

        mock_mkdir.side_effect = [PermissionError, OSError]

        with self.assertRaises(PermissionError):
            bkpt_folder, bkpt_file = filetools.choose_bkpt_file(self.testdir, round_=0)

        with self.assertRaises(OSError):
            bkpt_folder, bkpt_file = filetools.choose_bkpt_file(self.testdir, round_=0)

    def test_filetools_04_create_unique_link(self):
        """
        Test method `create_unique_link`
        """

        # OK creating links
        link_prefix = 'link_name'
        link_suffix = '_tutu'

        link_path = filetools.create_unique_link(
            self.testdir,
            link_prefix,
            link_suffix,
            'any_target_is_ok')
        self.assertEqual(
            link_path,
            os.path.join(self.testdir, link_prefix + link_suffix))
        self.assertTrue(os.path.islink(link_path))

        for index in range(1, 4):
            link_path = filetools.create_unique_link(
                self.testdir,
                link_prefix,
                link_suffix,
                'any_target_is_ok'
            )
            self.assertEqual(
                link_path,
                os.path.join(self.testdir, link_prefix + '_' + str("{:02d}".format(index)) + link_suffix))
            self.assertTrue(os.path.islink(link_path))

        # KO cannot create link
        bkpt_folder = 'subdir'
        bkpt_folder_path = os.path.join(self.testdir, bkpt_folder)
        self.assertRaises(
            FileNotFoundError,
            filetools.create_unique_link,
            bkpt_folder_path,
            link_prefix,
            link_suffix,
            'a_target_name')

        os.makedirs(bkpt_folder_path, mode=0o500)
        self.assertRaises(
            PermissionError,
            filetools.create_unique_link,
            bkpt_folder_path,
            link_prefix,
            link_suffix,
            'a_target_name')
        os.chmod(bkpt_folder_path, 0o700)

    @patch('fedbiomed.researcher.filetools.create_unique_link')
    def test_filetools_05_create_unique_file_link(
            self,
            patch_create_ul):
        """
        Test `create_unique_file_link` method
        """

        def side_create_ul(bkpt_folder_path, link_src_prefix, link_src_postfix, link_target):
            return os.path.join(bkpt_folder_path, link_src_prefix + link_src_postfix)

        patch_create_ul.side_effect = side_create_ul

        # OK choosing link source in same dir with good name
        src_folder = self.testdir
        target_file_names = [
            'this_is_the_name.py',
            'another243.tutu',
            'my_dummy.1.2.3.ok',
            'another.name.with.dots'
        ]
        for name in target_file_names:
            target_file_path = os.path.join(self.testdir, name)

            link_path = filetools.create_unique_file_link(src_folder, target_file_path)

            self.assertEqual(link_path, target_file_path)

        # OK choosing link source in subdir with good name
        src_folder = os.path.join(self.testdir, 'yet_another_dir')
        os.makedirs(src_folder)
        target_file = 'this_name_is.good'
        target_file_path = os.path.join(self.testdir, target_file)

        link_path = filetools.create_unique_file_link(src_folder, target_file_path)

        self.assertEqual(link_path, os.path.join(src_folder, target_file))

        # KO choosing link source in same dir with bad name
        src_folder = self.testdir
        target_file_names = [
            'bad_name',
            'this_is_the_name.',
            '.alsobad',
            'not.a.good.1',
            'does_not.match_ok'
        ]
        for name in target_file_names:
            target_file_path = os.path.join(self.testdir, name)

            self.assertRaises(
                ValueError,
                filetools.create_unique_file_link,
                src_folder,
                target_file_path)

        # KO choosing a file path not in subfolder
        os.symlink('tutu', os.path.join(self.testdir, 'hanging_link'))
        file_paths = [
            os.path.join(self.testdir, 'not_existing_dir/src_link'),
            os.path.join(self.testdir, 'hanging_link/src_link'),
            os.path.join(self.testdir, 'file.ok'),
            os.path.join(self.testdir, 'file.ok')
        ]
        bkpt_folder_paths = [
            self.testdir,
            self.testdir,
            os.path.join(self.testdir, 'not_existing_dir'),
            os.path.join(self.testdir, 'hanging_link')
        ]

        for file, folder in zip(file_paths, bkpt_folder_paths):
            self.assertRaises(
                ValueError,
                filetools.create_unique_file_link,
                folder,
                file)

    def test_filetools_06_private_get_latest_file(self):
        """
        Tests if `_get_latest_file` returns more recent file
        """

        # test 1
        files = ["Experiment_0",
                 "Experiment_4",
                 "Experiment_5",
                 "blabla",
                 "99_blabla"]

        pathfile_test = "/path/to/a/file"

        latest_file = filetools._get_latest_file(pathfile_test,
                                                 files,
                                                 only_folder=False)
        self.assertEqual(files[2], latest_file)

        # test 2: in this test, we patch isdir builtin function
        # so it returns always `True` when called
        patcher_builtin_os_path_isdir = patch("os.path.isdir",
                                              return_value=True)
        patcher_builtin_os_path_isdir.start()
        latest_file = filetools._get_latest_file(pathfile_test,
                                                 files,
                                                 only_folder=True)
        self.assertEqual(files[2], latest_file)
        patcher_builtin_os_path_isdir.stop()

        # test 3: file provided are not directory, raise exception
        self.assertRaises(FileNotFoundError,
                          filetools._get_latest_file,
                          pathfile_test,
                          files,
                          only_folder=True)

        # test 4: no file provided, raises exception
        files = []
        self.assertRaises(FileNotFoundError,
                          filetools._get_latest_file,
                          pathfile_test,
                          files,
                          only_folder=False)

        # test 5: test if exception is raised because no matching file
        # (ie finishing with numbers)
        files = ['q', 'foo', 'bar']

        self.assertRaises(FileNotFoundError,
                          filetools._get_latest_file,
                          pathfile_test,
                          files,
                          only_folder=False)

    @patch('os.path.isdir')
    @patch('os.listdir')
    def test_filetools_07_private_find_breakpoint_path_1(self,
                                                         patch_os_listdir,
                                                         patch_os_path_isdir
                                                         ):
        # test 1 : test if results are corrects  if path
        # to breakpoint has been given by user
        bkpt_folder = "/path/to/breakpoint"
        patch_os_listdir.return_value = ['breakpoint_1234.json',
                                         "another_file"]
        patch_os_path_isdir.return_value = True

        bkpt_folder_out, state_file = filetools.find_breakpoint_path(bkpt_folder)
        self.assertEqual(bkpt_folder, bkpt_folder_out)
        self.assertEqual(state_file, 'breakpoint_1234.json')

    @patch('fedbiomed.researcher.filetools._get_latest_file')
    @patch('os.path.isdir')
    @patch('os.listdir')
    def test_filetools_08_private_find_breakpoint_path_2(self,
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
        latest_bkpt_folder = os.path.join(environ['EXPERIMENTS_DIR'],
                                          'breakpoint',
                                          'breakpoint')

        bkpt_folder_out, state_file = filetools.find_breakpoint_path(None)
        self.assertEqual(state_file, 'breakpoint_1234.json')
        self.assertEqual(bkpt_folder_out, latest_bkpt_folder)

    @patch('os.path.isdir')
    @patch('os.listdir')
    def test_filetools_09_private_find_breakpoint_path_raise_err1(self,
                                                                  patch_os_listdir,
                                                                  patch_os_path_isdir):
        # test 3 : triggers error FileNotFoundError, given breakpoint folder
        # is not a directory
        bkpt_folder = "/path/to/breakpoint"
        patch_os_listdir.return_value = ['breakpoint_1234.json',
                                         "another_file"]
        patch_os_path_isdir.return_value = False
        self.assertRaises(FileNotFoundError,
                          filetools.find_breakpoint_path,
                          bkpt_folder)

    @patch('os.path.isfile')
    @patch('os.path.isdir')
    @patch('os.listdir')
    def test_filetools_10_private_find_breakpoint_path_raise_err2(self,
                                                                  patch_os_listdir,
                                                                  patch_os_path_isdir,
                                                                  patch_os_path_isfile):
        # test 4 : triggers error FileNotFoundError, no given breakpoint
        # folder, cannot guess one from existing folders
        patch_os_listdir.return_value = ['breakpoint_1234.json',
                                         "another_file"]
        patch_os_path_isdir.return_value = True
        patch_os_path_isfile.return_value = True
        self.assertRaises(FileNotFoundError,
                          filetools.find_breakpoint_path)

    def test_filetools_11_private_find_breakpoint_raise_err_3(self):
        # test 5 : triggers error FileNotFoundError, cannot guess a folder
        # when none exist.
        self.assertRaises(FileNotFoundError,
                          filetools.find_breakpoint_path,
                          None)

    @patch('os.path.isdir')
    @patch('os.listdir')
    def test_filetools_12_private_find_breakpoint_path_raise_err4(self,
                                                                  patch_os_listdir,
                                                                  patch_os_path_isdir):
        # test 6 : triggers error FileNotFoundError, no file
        # present in breakpoint folder
        bkpt_folder = "/path/to/breakpoint"
        patch_os_listdir.return_value = []
        patch_os_path_isdir.return_value = True
        self.assertRaises(FileNotFoundError,
                          filetools.find_breakpoint_path,
                          bkpt_folder)

    @patch('os.path.isdir')
    @patch('os.listdir')
    def test_filetools_13_private_find_breakpoint_raise_err_5(self,
                                                              patch_os_listdir,
                                                              patch_os_path_isdir):
        # test 7: test if error is raised when json file
        # not found in a breakpoint folder specified by user
        bkpt_folder = "/path/to/breakpoint"
        patch_os_listdir.return_value = ['one_file',
                                         "another_file"]
        patch_os_path_isdir.return_value = True

        self.assertRaises(FileNotFoundError,
                          filetools.find_breakpoint_path,
                          bkpt_folder)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
