import unittest
from unittest.mock import patch

import testsupport.mock_node_environ  ## noqa (remove flake8 false warning)

from testsupport.fake_message import FakeMessages

from fedbiomed.common.constants import SecaggElementTypes
from fedbiomed.common.exceptions import FedbiomedSecaggError
from fedbiomed.node.environ import environ
from fedbiomed.node.secagg import SecaggServkeySetup, SecaggBiprimeSetup


class TestSecaggResearcher(unittest.TestCase):
    """ Test for researcher's secagg module"""

    @classmethod
    def setUpClass(cls) -> None:
        pass

    @classmethod
    def tearDownClass(cls) -> None:
        pass

    def setUp(self):
        pass

    def tearDown(self) -> None:
        pass

    def test_secagg_01_init_ok_and_getters(self):
        """Instantiate secagg classes + read state via getters"""

        # prepare
        kwargs = {
            'researcher_id': "my researcher",
            'secagg_id': "my secagg",
            'sequence': 123,
            'parties': ['my researcher', 'my node1', 'my node2', 'my node3'],
        }
        secagg_setups = [
            [SecaggServkeySetup, SecaggElementTypes.SERVER_KEY],
            [SecaggBiprimeSetup, SecaggElementTypes.BIPRIME],
        ]

        for secagg_setup, element_type in secagg_setups :
            # test
            secagg = secagg_setup(**kwargs)

            # check
            self.assertEqual(secagg.researcher_id(), kwargs['researcher_id'])
            self.assertEqual(secagg.secagg_id(), kwargs['secagg_id'])
            self.assertEqual(secagg.sequence(), kwargs['sequence'])
            self.assertEqual(secagg.element(), element_type)

    def test_secagg_02_init_badargs(self):
        """Instantiate secagg classes with bad arguments"""

        # prepare
        kwargs_list = [
            {
                'researcher_id': None,
                'secagg_id': "my secagg",
                'sequence': 123,
                'parties': ['my researcher', 'my node1', 'my node2', 'my node3'],
            },
            {
                'researcher_id': 234,
                'secagg_id': "my secagg",
                'sequence': 123,
                'parties': ['my researcher', 'my node1', 'my node2', 'my node3'],
            },
            {
                'researcher_id': "",
                'secagg_id': "my secagg",
                'sequence': 123,
                'parties': ['my researcher', 'my node1', 'my node2', 'my node3'],
            },
            {
                'researcher_id': 'my researcher',
                'secagg_id': None,
                'sequence': 123,
                'parties': ['my researcher', 'my node1', 'my node2', 'my node3'],
            },
            {
                'researcher_id': 'my researcher',
                'secagg_id': 12345,
                'sequence': 123,
                'parties': ['my researcher', 'my node1', 'my node2', 'my node3'],
            },
            {
                'researcher_id': 'my researcher',
                'secagg_id': "",
                'sequence': 123,
                'parties': ['my researcher', 'my node1', 'my node2', 'my node3'],
            },
            {
                'researcher_id': 'my researcher',
                'secagg_id': "my secagg",
                'sequence': None,
                'parties': ['my researcher', 'my node1', 'my node2', 'my node3'],
            },
            {
                'researcher_id': 'my researcher',
                'secagg_id': "my secagg",
                'sequence': 'sequence is not a string',
                'parties': ['my researcher', 'my node1', 'my node2', 'my node3'],
            },
            {
                'researcher_id': 'my researcher',
                'secagg_id': "my secagg",
                'sequence': ['sequence is not a list'],
                'parties': ['my researcher', 'my node1', 'my node2', 'my node3'],
            },
            {
                'researcher_id': 'my researcher',
                'secagg_id': "my secagg",
                'sequence': 123,
                'parties': None,
            },
            {
                'researcher_id': 'my researcher',
                'secagg_id': "my secagg",
                'sequence': 123,
                'parties': 'need to be a list',
            },
            {
                'researcher_id': 'my researcher',
                'secagg_id': "my secagg",
                'sequence': 123,
                'parties': [None, None],
            },
            {
                'researcher_id': 'my researcher',
                'secagg_id': "my secagg",
                'sequence': 123,
                'parties': [654, 321],
            },
            {
                'researcher_id': 'my researcher',
                'secagg_id': "my secagg",
                'sequence': 123,
                'parties': ['need to be same as researcher_id', 'my node2', 'my node3'],
            },
            {
                'researcher_id': 'my researcher',
                'secagg_id': "my secagg",
                'sequence': 123,
                'parties': ['my researcher', 'need 3+ parties'],
            },
        ]
        secagg_setups = [SecaggServkeySetup, SecaggBiprimeSetup]

        for kwargs in kwargs_list:
            for secagg_setup in secagg_setups :
                # test
                with self.assertRaises(FedbiomedSecaggError):
                    secagg_setup(**kwargs)


    @patch('time.sleep')
    @patch('fedbiomed.node.secagg.NodeMessages.reply_create')
    def test_secagg_03_setup(
            self,
            patch_reply_create,
            patch_time_sleep):
        """Setup secagg context elements
        """
        # patch
        # nota: dont need to patch time.sleep (dummy function is ok)
        def reply_create_side_effect(msg):
            return FakeMessages(msg)
        patch_reply_create.side_effect = reply_create_side_effect

        # prepare
        kwargs = {
            'researcher_id': "my researcher",
            'secagg_id': "my secagg",
            'sequence': 123,
            'parties': ['my researcher', 'my node1', 'my node2', 'my node3'],
        }
        secagg_setups = [
            SecaggServkeySetup,
            SecaggBiprimeSetup,
        ]

        for secagg_setup in secagg_setups :
            # test
            secagg = secagg_setup(**kwargs)
            msg = secagg.setup()

            # check
            self.assertEqual(msg['researcher_id'], kwargs['researcher_id'])
            self.assertEqual(msg['secagg_id'], kwargs['secagg_id'])
            self.assertEqual(msg['sequence'], kwargs['sequence'])
            self.assertEqual(msg['node_id'], environ['NODE_ID'])
            self.assertEqual(msg['success'], True)
            self.assertEqual(msg['command'], 'secagg')

    @patch('fedbiomed.node.secagg.NodeMessages.reply_create')
    def test_secagg_04_create_secagg_reply(
            self,
            patch_reply_create):
        """Create a reply message for researcher
        """
        # patch
        # nota: dont need to patch time.sleep (dummy function is ok)
        def reply_create_side_effect(msg):
            return FakeMessages(msg)
        patch_reply_create.side_effect = reply_create_side_effect

        # prepare
        kwargs = {
            'researcher_id': "my researcher",
            'secagg_id': "my secagg",
            'sequence': 123,
            'parties': ['my researcher', 'my node1', 'my node2', 'my node3'],
        }
        secagg_setups = [
            SecaggServkeySetup,
            SecaggBiprimeSetup,
        ]
        reply_message_list = [ '', 'custom reply message']
        reply_status_list = [False, True]

        for secagg_setup in secagg_setups :
            for reply_message in reply_message_list:
                for reply_status in reply_status_list:
                    # test
                    secagg = secagg_setup(**kwargs)
                    msg = secagg._create_secagg_reply(reply_message, reply_status)

                    # check
                    self.assertEqual(msg['researcher_id'], kwargs['researcher_id'])
                    self.assertEqual(msg['secagg_id'], kwargs['secagg_id'])
                    self.assertEqual(msg['sequence'], kwargs['sequence'])
                    self.assertEqual(msg['node_id'], environ['NODE_ID'])
                    self.assertEqual(msg['success'], reply_status)
                    self.assertEqual(msg['msg'], reply_message)
                    self.assertEqual(msg['command'], 'secagg')



if __name__ == '__main__':  # pragma: no cover
    unittest.main()
