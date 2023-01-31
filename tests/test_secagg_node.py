import unittest
from unittest.mock import patch
from copy import deepcopy

#############################################################
# Import NodeTestCase before importing FedBioMed Module
from testsupport.base_case import NodeTestCase
#############################################################

from testsupport.fake_message import FakeMessages
from testsupport.fake_secagg_manager import FakeSecaggServkeyManager, FakeSecaggBiprimeManager

from fedbiomed.common.constants import SecaggElementTypes
from fedbiomed.common.exceptions import FedbiomedSecaggError
from fedbiomed.node.environ import environ
from fedbiomed.node.secagg import SecaggServkeySetup, SecaggBiprimeSetup


class TestSecaggNode(NodeTestCase):
    """ Test for node's secagg module"""

    @patch('fedbiomed.node.secagg.SecaggBiprimeManager')
    @patch('fedbiomed.node.secagg.SecaggServkeyManager')
    def test_secagg_01_init_ok_and_getters(
            self,
            patch_servkey_manager,
            patch_biprime_manager):
        """Instantiate secagg classes + read state via getters"""

        # prepare
        patch_servkey_manager.return_value = FakeSecaggServkeyManager()
        patch_biprime_manager.return_value = FakeSecaggBiprimeManager()

        kwargs_servkey = {
            'researcher_id': "my researcher",
            'secagg_id': "my secagg",
            'job_id': 'my_job_id',
            'sequence': 123,
            'parties': ['my researcher', 'my node1', 'my node2', 'my node3'],
        }
        kwargs_biprime = deepcopy(kwargs_servkey)
        kwargs_biprime['job_id'] = ''
        secagg_setups = [
            [SecaggServkeySetup, SecaggElementTypes.SERVER_KEY, kwargs_servkey],
            [SecaggBiprimeSetup, SecaggElementTypes.BIPRIME, kwargs_biprime],
        ]

        for secagg_setup, element_type, kwargs in secagg_setups :
            # test
            secagg = secagg_setup(**kwargs)

            # check
            self.assertEqual(secagg.researcher_id(), kwargs['researcher_id'])
            self.assertEqual(secagg.secagg_id(), kwargs['secagg_id'])
            self.assertEqual(secagg.job_id(), kwargs['job_id'])
            self.assertEqual(secagg.sequence(), kwargs['sequence'])
            self.assertEqual(secagg.element(), element_type)

    @patch('fedbiomed.node.secagg.SecaggBiprimeManager')
    @patch('fedbiomed.node.secagg.SecaggServkeyManager')
    def test_secagg_02_init_badargs(
            self,
            patch_servkey_manager,
            patch_biprime_manager,
            ):
        """Instantiate secagg classes with bad arguments"""

        # prepare
        patch_servkey_manager.return_value = FakeSecaggServkeyManager()
        patch_biprime_manager.return_value = FakeSecaggBiprimeManager()

        job_id = 'my_job'
        kwargs_servkey_list = [
            {
                'researcher_id': None,
                'secagg_id': "my secagg",
                'job_id': job_id,
                'sequence': 123,
                'parties': ['my researcher', 'my node1', 'my node2', 'my node3'],
            },
            {
                'researcher_id': 234,
                'secagg_id': "my secagg",
                'job_id': job_id,
                'sequence': 123,
                'parties': ['my researcher', 'my node1', 'my node2', 'my node3'],
            },
            {
                'researcher_id': "",
                'secagg_id': "my secagg",
                'job_id': job_id,
                'sequence': 123,
                'parties': ['my researcher', 'my node1', 'my node2', 'my node3'],
            },
            {
                'researcher_id': 'my researcher',
                'secagg_id': None,
                'job_id': job_id,
                'sequence': 123,
                'parties': ['my researcher', 'my node1', 'my node2', 'my node3'],
            },
            {
                'researcher_id': 'my researcher',
                'secagg_id': 12345,
                'job_id': job_id,
                'sequence': 123,
                'parties': ['my researcher', 'my node1', 'my node2', 'my node3'],
            },
            {
                'researcher_id': 'my researcher',
                'secagg_id': "",
                'job_id': job_id,
                'sequence': 123,
                'parties': ['my researcher', 'my node1', 'my node2', 'my node3'],
            },
            {
                'researcher_id': 'my researcher',
                'secagg_id': "my secagg",
                'job_id': ['not a string'],
                'sequence': 123,
                'parties': ['my researcher', 'my node1', 'my node2', 'my node3'],
            },
            {
                'researcher_id': 'my researcher',
                'secagg_id': "my secagg",
                'job_id': 999,
                'sequence': 123,
                'parties': ['my researcher', 'my node1', 'my node2', 'my node3'],
            },
            {
                'researcher_id': 'my researcher',
                'secagg_id': "my secagg",
                'job_id': '',
                'sequence': 123,
                'parties': ['my researcher', 'my node1', 'my node2', 'my node3'],
            },
            {
                'researcher_id': 'my researcher',
                'secagg_id': "my secagg",
                'job_id': job_id,
                'sequence': None,
                'parties': ['my researcher', 'my node1', 'my node2', 'my node3'],
            },
            {
                'researcher_id': 'my researcher',
                'secagg_id': "my secagg",
                'job_id': job_id,
                'sequence': 'sequence is not a string',
                'parties': ['my researcher', 'my node1', 'my node2', 'my node3'],
            },
            {
                'researcher_id': 'my researcher',
                'secagg_id': "my secagg",
                'job_id': job_id,
                'sequence': ['sequence is not a list'],
                'parties': ['my researcher', 'my node1', 'my node2', 'my node3'],
            },
            {
                'researcher_id': 'my researcher',
                'secagg_id': "my secagg",
                'job_id': job_id,
                'sequence': 123,
                'parties': None,
            },
            {
                'researcher_id': 'my researcher',
                'secagg_id': "my secagg",
                'job_id': job_id,
                'sequence': 123,
                'parties': 'need to be a list',
            },
            {
                'researcher_id': 'my researcher',
                'secagg_id': "my secagg",
                'job_id': job_id,
                'sequence': 123,
                'parties': [None, None],
            },
            {
                'researcher_id': 'my researcher',
                'secagg_id': "my secagg",
                'job_id': job_id,
                'sequence': 123,
                'parties': [654, 321],
            },
            {
                'researcher_id': 'my researcher',
                'secagg_id': "my secagg",
                'job_id': job_id,
                'sequence': 123,
                'parties': ['need to be same as researcher_id', 'my node2', 'my node3'],
            },
            {
                'researcher_id': 'my researcher',
                'secagg_id': "my secagg",
                'job_id': job_id,
                'sequence': 123,
                'parties': ['my researcher', 'need 3+ parties'],
            },
        ]
        kwargs_biprime_list = deepcopy(kwargs_servkey_list)
        # normal case for servkey (non empty `job_id` string) is the error case for biprime (and vice verse)
        for kwargs in kwargs_biprime_list:
            if kwargs['job_id'] == job_id:
                kwargs['job_id'] = ''
            elif kwargs['job_id'] == '':
                kwargs['job_id'] = job_id

        secaggs = [
            (SecaggServkeySetup, kwargs_servkey_list),
            (SecaggBiprimeSetup, kwargs_biprime_list)
        ]

        for secagg_setup, kwargs_list in secaggs :
            for kwargs in kwargs_list:
                # test
                with self.assertRaises(FedbiomedSecaggError):
                    secagg_setup(**kwargs)


    @patch('time.sleep')
    @patch('fedbiomed.node.secagg.SecaggBiprimeManager')
    @patch('fedbiomed.node.secagg.SecaggServkeyManager')
    @patch('fedbiomed.node.secagg.NodeMessages.reply_create')
    def test_secagg_03_setup(
            self,
            patch_reply_create,
            patch_servkey_manager,
            patch_biprime_manager,
            patch_time_sleep):
        """Setup secagg context elements
        """
        # patch
        # nota: dont need to patch time.sleep (dummy function is ok)
        def reply_create_side_effect(msg):
            return FakeMessages(msg)
        patch_reply_create.side_effect = reply_create_side_effect

        patch_servkey_manager.return_value = FakeSecaggServkeyManager()
        patch_biprime_manager.return_value = FakeSecaggBiprimeManager()

        # prepare
        kwargs_servkey = {
            'researcher_id': "my researcher",
            'secagg_id': "my secagg",
            'job_id': "my_job",
            'sequence': 123,
            'parties': ['my researcher', 'my node1', 'my node2', 'my node3'],
        }
        kwargs_biprime = deepcopy(kwargs_servkey)
        kwargs_biprime['job_id'] = ''
        secagg_setups = [
            (SecaggServkeySetup, kwargs_servkey),
            (SecaggBiprimeSetup, kwargs_biprime),
        ]

        for secagg_setup, kwargs in secagg_setups :
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

    @patch('fedbiomed.node.secagg.SecaggBiprimeManager')
    @patch('fedbiomed.node.secagg.SecaggServkeyManager')
    @patch('fedbiomed.node.secagg.NodeMessages.reply_create')
    def test_secagg_04_create_secagg_reply(
            self,
            patch_reply_create,
            patch_servkey_manager,
            patch_biprime_manager):
        """Create a reply message for researcher
        """
        # patch
        # nota: dont need to patch time.sleep (dummy function is ok)
        def reply_create_side_effect(msg):
            return FakeMessages(msg)
        patch_reply_create.side_effect = reply_create_side_effect

        patch_servkey_manager.return_value = FakeSecaggServkeyManager()
        patch_biprime_manager.return_value = FakeSecaggBiprimeManager()

        # prepare
        kwargs_servkey = {
            'researcher_id': "my researcher",
            'secagg_id': "my secagg",
            'job_id': "my_job",
            'sequence': 123,
            'parties': ['my researcher', 'my node1', 'my node2', 'my node3'],
        }
        kwargs_biprime = deepcopy(kwargs_servkey)
        kwargs_biprime['job_id'] = ''
        secagg_setups = [
            (SecaggServkeySetup, kwargs_servkey),
            (SecaggBiprimeSetup, kwargs_biprime),
        ]
        reply_message_list = [ '', 'custom reply message']
        reply_status_list = [False, True]

        for secagg_setup, kwargs in secagg_setups :
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
