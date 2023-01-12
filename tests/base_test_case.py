import shutil
import unittest

from unittest.mock import patch
from testsupport.fake_environ import ResearcherEnviron, NodeEnviron


class ResearcherTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.env = ResearcherEnviron()

        def side_effect(item):
            return cls.env[item]

        cls.environ_patch = patch("fedbiomed.node.environ.ResearcherEnviron.__getitem__")
        cls.environ = cls.environ_patch.start()
        cls.environ.side_effect = side_effect
        pass

    @classmethod
    def tearDownClass(cls) -> None:
        # Clean environ
        shutil.rmtree(cls.env["ROOT_DIR"])
        cls.environ_patch.stop()
    pass


class NodeTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.env = NodeEnviron()

        def side_effect(item):
            return cls.env[item]

        cls.environ_patch = patch("fedbiomed.node.environ.NodeEnviron.__getitem__")
        cls.environ = cls.environ_patch.start()
        cls.environ.side_effect = side_effect
        pass

    @classmethod
    def tearDownClass(cls) -> None:
        # Clean environ
        shutil.rmtree(cls.env["ROOT_DIR"])
        cls.environ_patch.stop()


