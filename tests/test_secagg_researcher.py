import copy
import unittest
from unittest.mock import patch

#############################################################
# Import ResearcherTestCase before importing any FedBioMed Module
from testsupport.base_case import ResearcherTestCase
#############################################################

from testsupport.fake_requests import FakeRequests
from fedbiomed.common.exceptions import FedbiomedSecaggError
from fedbiomed.researcher.secagg import SecaggServkeyContext, SecaggBiprimeContext, SecaggContext


class TestSecaggResearcher(ResearcherTestCase):
    """ Test for researcher's secagg module"""

    def setUp(self):
        # Define patchers that are not modified during the test
        self.patchers = [
            patch('fedbiomed.researcher.requests.Requests.__init__',
                  return_value=None),
        ]
        # Note: we could mock Validate, but we considered OK to use the real class

        for patcher in self.patchers:
            patcher.start()

        print(self.env["RESEARCHER_ID"])

    def tearDown(self) -> None:
        for patcher in self.patchers:
            patcher.stop()

    def test_secagg_01_init(self):
        """Instantiate secagg classes"""

        # Correct secagg classes instantiations

        #
        parties_set = [
            [ '', '', ''],
            [ 'party1', 'party2', 'party3'],
            [ 'party1', 'party2', 'party3', 'party4'],
            [ 'party1', 'party2', 'party3', 'party4', 'party5', 'party6', 'party7', 'party8', 'party9', 'party10'],
        ]
        job_ids = ['JOB_ID', 'another string but not empty']
        for parties in parties_set:
            for job_id in job_ids:
                # test
                servkey = SecaggServkeyContext(parties, job_id)
                biprime = SecaggBiprimeContext(parties)

                # check
                self.assertEqual(servkey.context(), None)
                self.assertTrue(isinstance(servkey.secagg_id(), str))
                self.assertFalse(servkey.status())
                self.assertEqual(biprime.context(), None)
                self.assertTrue(isinstance(biprime.secagg_id(), str))
                self.assertFalse(biprime.status())

        # Bad secagg classes instantiations

        # prepare
        parties_set = [
            [],
            [ 'party1'],
            [ 'party1', 'party2'],
            [ 'party1', 'party2', None],
            [ 'party1', 'party2', True],
            [ 'party1', 'party2', 3],
            [ 'party1', 'party2', ['party3']],            
        ]       
        for parties in parties_set:
            # check
            with self.assertRaises(FedbiomedSecaggError):
                SecaggServkeyContext(parties, 'JOB_ID')
            with self.assertRaises(FedbiomedSecaggError):
                SecaggBiprimeContext(parties)            

        job_id_set = [
            '',
            ['not a string but an array'],
            123,
            None,
            True,
            {'not a string'},
        ]
        for job_id in job_id_set:
            # check
            with self.assertRaises(FedbiomedSecaggError):
                SecaggServkeyContext(['p1', 'p2', 'p3'], job_id)            

    def test_secagg_02_dummy_abstract(self):
        """Dummy test for abstract methods"""

        # Want a dummy test for abstract method (unused) code to keep code coverage
        patcher = patch.multiple(SecaggContext, __abstractmethods__=set())
        patcher.start()
        dummy = SecaggContext(['un', 'deux', 'trois'], 'JOB_ID')
        dummy._payload()
        patcher.stop()

        # no check, just for coverage

    @patch('time.sleep')
    @patch('fedbiomed.researcher.requests.Requests.send_message')
    @patch('fedbiomed.researcher.requests.Requests.get_responses')
    def test_secagg_03_setup_delete(
            self,
            patch_requests_get_responses,
            patch_requests_send_message,
            patch_time_sleep):
        """Setup then delete a secagg class"""

        # prepare
        parties = [self.env['RESEARCHER_ID'], 'party2', 'party3']
        job_id = 'JOB_ID'

        fake_requests = FakeRequests()
        patch_requests_send_message.side_effect = fake_requests.send_message
        patch_requests_get_responses.side_effect = fake_requests.get_responses

        #
        # 1. Correctly setup and delete
        #

        # time.sleep: just need a dummy patch to avoid waiting

        # test with no established context, then with already existing context
        for i in range(2):
            # test setup
            secagg = SecaggServkeyContext(parties, job_id)
            secagg.setup(timeout=5)
            biprime = SecaggBiprimeContext(parties)
            biprime.setup(timeout=5)

            # check setup
            self.assertTrue(secagg.status())
            self.assertEqual(secagg.context()['msg'], 'Not implemented yet')
            self.assertTrue(biprime.status())
            self.assertEqual(biprime.context()['msg'], 'Not implemented yet')

        # test on established context, then with no existing context
        for i in range(2):
            # test delete 
            secagg.delete(timeout=1)
            biprime.delete(timeout=1)

            # check delete 
            self.assertFalse(secagg.status())
            self.assertEqual(secagg.context(), None)
            self.assertFalse(biprime.status())
            self.assertEqual(biprime.context(), None)

        #
        # 2. Setup and delete with errors
        #

        # prepare
        customs = [
            {'researcher_id': 'ANOTHER_RESEARCHER'},
            {'secagg_id': 'ANOTHER_SECAGG'},
            {'node_id': 'ANOTHER_NODE'},
            {'sequence': 12345}
        ]
        contexts = [SecaggServkeyContext(parties, job_id), SecaggBiprimeContext(parties)]

        # test and check, on non-established then on established context
        for i in range(2):
            for context in contexts:
                if i == 1:
                    fake_requests.set_replies_custom_fields({})
                    context.setup()
                for custom in customs:
                    fake_requests.set_replies_custom_fields(custom)

                    # delete can fail only if there is a context to delete
                    if context.status():
                        with self.assertRaises(FedbiomedSecaggError):
                            context.delete(timeout=0.1)
                    # need to delete before failed setup (else, status is False)
                    with self.assertRaises(FedbiomedSecaggError):
                        context.setup(timeout=0.1)

    @patch('fedbiomed.researcher.requests.Requests.send_message')
    @patch('fedbiomed.researcher.requests.Requests.get_responses')
    def test_secagg_04_setup_delete_timeout(
            self,
            patch_requests_get_responses,
            patch_requests_send_message):
        """Timeout during secagg class setup"""

        # prepare
        parties = [self.env['RESEARCHER_ID'], 'party2', 'party3']
        job_id = 'JOD ID'

        fake_requests = FakeRequests()
        patch_requests_send_message.side_effect = fake_requests.send_message
        patch_requests_get_responses.side_effect = fake_requests.get_responses

        contexts = [SecaggServkeyContext(parties, job_id), SecaggBiprimeContext(parties)]
        timeouts = [0.1, 0.2, 0.45]

        # test and check
        for t in timeouts:
            for context in contexts:
                with self.assertRaises(FedbiomedSecaggError):
                    context.setup(timeout=t)

    def test_secagg_05_setup_delete_badparams(self):
        """Try setup or delete a secagg class giving bad params"""

        # setup
        parties = [ self.env['RESEARCHER_ID'], 'party2', 'party3']
        job_id = 'JOB ID'
        contexts = [SecaggServkeyContext(parties, job_id), SecaggBiprimeContext(parties)]
        values = ['2', '2.3', '', [2], {'3': 3}]

        # check and test
        for context in contexts:
            for value in values:
                with self.assertRaises(FedbiomedSecaggError):
                    context.setup(value)
                with self.assertRaises(FedbiomedSecaggError):
                    context.delete(value)

    @patch('time.sleep')
    @patch('fedbiomed.researcher.requests.Requests.send_message')
    @patch('fedbiomed.researcher.requests.Requests.get_responses')
    def test_secagg_06_breakpoint(
            self,
            patch_requests_get_responses,
            patch_requests_send_message,
            patch_time_sleep):
        """Save and load breakpoint status for secagg class"""

        # 1. Save breakpoint

        # prepare
        parties = [self.env['RESEARCHER_ID'], 'node1', 'node2', 'node3']
        job_id = 'JOB_ID'

        # time.sleep: just need a dummy patch to avoid waiting

        fake_requests = FakeRequests()
        patch_requests_send_message.side_effect = fake_requests.send_message
        patch_requests_get_responses.side_effect = fake_requests.get_responses

        contexts = [
            (SecaggServkeyContext(parties, job_id), False),
            (SecaggBiprimeContext(parties), True)
        ]

        # test with no context, then with established context
        for i in range(2):
            for context, empty_job_id in contexts:
                expected_state = {
                    'class': type(context).__name__,
                    'module': context.__module__,
                    'secagg_id': context.secagg_id(),
                    'job_id': job_id,
                    'parties': parties,
                    'researcher_id': self.env['RESEARCHER_ID'],
                    'status': context.status(),
                    'context': context.context(),
                }
                if empty_job_id:
                    expected_state['job_id'] = ''

                # test
                state = context.save_state()
                # check
                self.assertEqual(state, expected_state)

                context.setup()

        # 2. Load complete breakpoint
        # nota: Cannot test content of complete state (not verified by function)

        # prepare
        state = {
            'secagg_id': 'my_secagg_id',
            'parties': ['ONE_PARTY', 'TWO_PARTIES', 'THREE_PARTIES'],
            'researcher_id': 'my_researcher_id',
            'job_id': 'my_job_id',
            'status': False,
            'context': 'MY CONTEXT'
        }
        contexts = [SecaggServkeyContext(parties, 'ANY_JOB_ID'), SecaggBiprimeContext(parties)]

        # test with no context, then with established context
        for i in range(2):
            for context in contexts:
                if range == 1:
                    context.setup()

                # test
                context.load_state(state)
                # check
                self.assertEqual(state['status'], context.status())
                self.assertEqual(state['secagg_id'], context.secagg_id())
                self.assertEqual(state['job_id'], context.job_id())
                self.assertEqual(state['context'], context.context())

        # 3. Load incomplete breakpoints
        # nota: error not handled by Fed-BioMed for now
        for context in contexts:
            for k in state.keys():
                tempo_state = copy.deepcopy(state)
                del tempo_state[k]

                with self.assertRaises(KeyError):
                    context.load_state(tempo_state)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
