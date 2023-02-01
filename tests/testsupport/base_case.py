import sys
import os
import shutil
import unittest
import glob
import testsupport.fake_node_environ

from testsupport.fake_node_environ import NodeRandomEnv
from testsupport.fake_researcher_environ import ResearcherRandomEnv

from unittest.mock import patch

sys.modules['fedbiomed.node.environ'] = testsupport.fake_node_environ
sys.modules['fedbiomed.researcher.environ'] = testsupport.fake_researcher_environ


class BaseTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:

        cls.env = None
        cls.mock_is_setup = True
        cls.environ_patch = None
        cls.environ_set_patch = None

    @classmethod
    def tearDownClass(cls) -> None:

        if not hasattr(cls, "mock_is_setup") or cls.mock_is_setup is not True:
            raise Exception("It seems that you are using one of ResearcherTestCase or NodeTestCase by overwriting "
                            "`setUpClass`. Please call super().setUpClass() at the very beginning of the method "
                            "`setUpClass`.")

        shutil.rmtree(cls.env["ROOT_DIR"])

        # for item in glob.glob(os.path.join('/tmp', "_res_*")):
        #     if not os.path.isdir(item):
        #         continue
        #     shutil.rmtree(item)
        #
        # for item in glob.glob(os.path.join('/tmp', "_nod_*")):
        #     if not os.path.isdir(item):
        #         continue
        #     shutil.rmtree(item)

        cls.environ_patch.stop()
        cls.environ_set_patch.stop()


class ResearcherTestCase(BaseTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        cls.env = ResearcherRandomEnv()

        def side_effect(item):
            return cls.env[item]

        def set_side_effect(key, value):
            cls.env[key] = value

        sys.modules['fedbiomed.node.environ'] = testsupport.fake_node_environ
        sys.modules['fedbiomed.researcher.environ'] = testsupport.fake_researcher_environ

        cls.environ_patch = patch("testsupport.fake_researcher_environ.ResearcherEnviron.__getitem__")
        cls.environ_set_patch = patch("testsupport.fake_researcher_environ.ResearcherEnviron.__setitem__")

        cls.environ = cls.environ_patch.start()
        cls.environ_set = cls.environ_set_patch.start()

        cls.environ.side_effect = side_effect
        cls.environ_set.side_effect = set_side_effect


class NodeTestCase(BaseTestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        cls.env = NodeRandomEnv()

        def side_effect(item):
            return cls.env[item]

        def set_side_effect(key, value):
            cls.env[key] = value

        cls.environ_patch = patch("testsupport.fake_node_environ.NodeEnviron.__getitem__")
        cls.environ_set_patch = patch("testsupport.fake_node_environ.NodeEnviron.__setitem__")

        cls.environ = cls.environ_patch.start()
        cls.environ_set = cls.environ_set_patch.start()
        cls.environ.side_effect = side_effect
        cls.environ_set.side_effect = set_side_effect



