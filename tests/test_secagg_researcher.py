import copy
import unittest
from unittest.mock import patch

#############################################################
# Import ResearcherTestCase before importing any FedBioMed Module
from testsupport.base_case import ResearcherTestCase
#############################################################
import fedbiomed.researcher.secagg

from fedbiomed.researcher.environ import environ
from testsupport.fake_requests import FakeRequests
from fedbiomed.common.exceptions import FedbiomedSecaggError
from fedbiomed.common.constants import SecaggElementTypes
from fedbiomed.researcher.secagg import SecaggServkeyContext, SecaggBiprimeContext, SecaggContext
from fedbiomed.researcher.responses import Responses

class BaseTestCaseSecaggContext(ResearcherTestCase):

    def setUp(self) -> None:
        self.patch_cm = patch.object(fedbiomed.researcher.secagg, "_CManager")
        self.patch_mpc = patch.object(fedbiomed.researcher.secagg, "MPCController")
        self.patch_requests = patch("fedbiomed.researcher.secagg.Requests")
        self.patch_skmanager = patch.object(fedbiomed.researcher.secagg, "_SKManager")
        self.patch_bpmanager = patch.object(fedbiomed.researcher.secagg, "_BPrimeManager")

        self.mock_cm = self.patch_cm.start()
        self.mock_mpc = self.patch_mpc.start()
        self.m_requests = self.patch_requests.start()
        self.mock_skmanager = self.patch_skmanager.start()
        self.mock_bpmanager = self.patch_bpmanager.start()

        # Set MOCK variables
        self.mock_cm.write_mpc_certificates_for_experiment.return_value = ('dummy/ip', [])
        self.mock_mpc.exec_shamir.return_value = 'dummy/path/to/output'
        unittest.mock.MagicMock.mpc_data_dir = unittest.mock.PropertyMock(
            return_value='dummy/path/to/output'
        )
        unittest.mock.MagicMock.tmp_dir = unittest.mock.PropertyMock(
            return_value=environ["TMP_DIR"]
        )

    def tearDown(self) -> None:
        self.patch_cm.stop()
        self.patch_mpc.stop()
        self.patch_requests.stop()
        self.patch_skmanager.stop()
        self.patch_bpmanager.stop()


class TestBaseSecaggContext(BaseTestCaseSecaggContext):

    def setUp(self):
        super().setUp()
        self.abstract_methods_patcher = patch.multiple(SecaggContext, __abstractmethods__=set())
        self.abstract_methods_patcher.start()
        self.secagg_context = SecaggContext(parties=[environ["ID"], 'party2', 'party3'],
                                            job_id="job-id")

    def tearDown(self) -> None:
        super().tearDown()
        self.abstract_methods_patcher.stop()

    def test_base_secagg_context_01_init(self):
        with self.assertRaises(FedbiomedSecaggError):
            SecaggContext(parties=['party1', 'party2', 'party3'],
                          job_id="job-id")

        # Less than 3 parties
        with self.assertRaises(FedbiomedSecaggError):
            SecaggContext(parties=[environ["ID"], 'party2'],
                          job_id="job-id")

        # Invalid type parties
        with self.assertRaises(FedbiomedSecaggError):
            SecaggContext(parties=[environ["ID"], 12, 12],
                          job_id="job-id")

        # Job id is not string
        with self.assertRaises(FedbiomedSecaggError):
            SecaggContext(parties=[environ["ID"], 'party2', 'party3'],
                          job_id=111)

    def test_secagg_context_02_getters_setters(self):
        """Tests setters and getters """

        self.assertEqual(self.secagg_context.job_id, "job-id")
        self.assertIsInstance(self.secagg_context.secagg_id, str)
        self.assertFalse(self.secagg_context.status)
        self.assertIsNone(self.secagg_context.context)

        self.secagg_context.set_job_id("new-job-id")
        self.assertEqual(self.secagg_context.job_id, "new-job-id")

        with self.assertRaises(FedbiomedSecaggError):
            self.secagg_context.set_job_id(1111)

    @patch('time.sleep')
    def test_secagg_context_03_secagg_round(
            self,
            patch_time_sleep):
        """Setup then delete a secagg class"""

        reply = {'researcher_id': environ["ID"],
                 'secagg_id': self.secagg_context.secagg_id,
                 'sequence': 123,
                 'success': True,
                 'node_id': "party2",
                 'msg': 'Fake request',
                 'command': 'secagg'}

        reply2 = copy.deepcopy(reply)
        reply2["node_id"] = "party3"

        # Patch response
        self.m_requests.return_value.get_responses.return_value = Responses([reply, reply2])
        self.m_requests.return_value.send_message.return_value = 123
        self.secagg_context._element = SecaggElementTypes.SERVER_KEY

        with patch("fedbiomed.researcher.secagg.SecaggContext._payload") as mock_payload:
            mock_payload.return_value = ("KEY", True)
            result = self.secagg_context.setup(timeout=1)
            self.assertTrue(result)

            with self.assertRaises(FedbiomedSecaggError):
                reply2["node_id"] = "party5"
                self.secagg_context.setup(timeout=1)

            with self.assertRaises(FedbiomedSecaggError):
                reply2["researcher_id"] = "party556"
                self.secagg_context.setup(timeout=1)

            with self.assertRaises(FedbiomedSecaggError):
                reply2["researcher_id"] = "party556"
                self.secagg_context.setup(timeout="!23")

            with self.assertRaises(FedbiomedSecaggError):
                reply2["researcher_id"] = environ["ID"]
                reply2["node_id"] = "party3"
                reply2["sequence"] = 000
                self.secagg_context.setup(timeout=1)

            with self.assertRaises(FedbiomedSecaggError):
                reply2["secagg_id"] = "non-id"
                reply2["sequence"] = 123
                self.secagg_context.setup(timeout=1)

    def test_secagg_06_breakpoint(
            self):
        """Save and load breakpoint status for secagg class"""

        expected_state = {
            'class': type(self.secagg_context).__name__,
            'module': self.secagg_context.__module__,
            'secagg_id': self.secagg_context.secagg_id,
            'job_id': self.secagg_context.job_id,
            'parties': self.secagg_context.parties,
            'researcher_id': self.env['RESEARCHER_ID'],
            'status': self.secagg_context.status,
            'context': self.secagg_context.context,
        }

        state = self.secagg_context.save_state()
        self.assertEqual(state, expected_state)

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

        self.secagg_context.load_state(state)

        self.assertEqual(state['status'], self.secagg_context.status)
        self.assertEqual(state['secagg_id'], self.secagg_context.secagg_id)
        self.assertEqual(state['job_id'], self.secagg_context.job_id)
        self.assertEqual(state['context'], self.secagg_context.context)


class TestSecaggServkeyContext(BaseTestCaseSecaggContext):

    def setUp(self) -> None:
        super().setUp()

        self.mock_skmanager.get.return_value = None
        self.srvkey_context = SecaggServkeyContext(parties=[environ["ID"], 'party2', 'party3'],
                                                   job_id="job-id")
        pass

    def tearDown(self) -> None:
        super().tearDown()
        pass

    def test_servkey_context_01_payload(self):
        """Tests failed init due to job id"""

        with self.assertRaises(FedbiomedSecaggError):
            SecaggServkeyContext(parties=[environ["ID"], 'party2', 'party3'],
                                 job_id=None)

        with self.assertRaises(FedbiomedSecaggError):
            SecaggServkeyContext(parties=[environ["ID"], 'party2', 'party3'],
                                 job_id="")

    def test_servkey_context_02_payload(self):
        with patch("builtins.open") as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = "key"

            context, status = self.srvkey_context._payload()

            self.mock_cm.write_mpc_certificates_for_experiment.assert_called_once_with(
                path_certificates='dummy/path/to/output',
                path_ips=environ["TMP_DIR"],
                self_id=environ["ID"],
                self_ip=environ["MPSPDZ_IP"],
                self_port=environ["MPSPDZ_PORT"],
                self_private_key=environ["MPSPDZ_CERTIFICATE_KEY"],
                self_public_key=environ["MPSPDZ_CERTIFICATE_PEM"],
                parties=[environ["ID"], 'party2', 'party3']
            )

            self.mock_mpc.exec_shamir.called_once_with(
                party_number=0,
                num_parties=3,
                ip_addresses='dummy/ip'
            )

            self.assertEqual(context['server_key'], "key")
            self.assertEqual(status, True)

            mock_open.side_effect = Exception
            with self.assertRaises(FedbiomedSecaggError):
                self.srvkey_context._payload()

            # note: should not be accessing private `_MPC` of `SecaggServkeyContext`
            # but could not have it working "clean" mocking
            self.srvkey_context._MPC.exec_shamir.side_effect = Exception
            with self.assertRaises(FedbiomedSecaggError):
                self.srvkey_context._payload()

    @patch('time.sleep')
    def test_servkey_context_03_secagg_delete(
            self,
            patch_time_sleep):
        """Setup then delete a secagg class"""

        reply = {'researcher_id': environ["ID"],
                 'secagg_id': self.srvkey_context.secagg_id,
                 'sequence': 123,
                 'success': True,
                 'node_id': "party2",
                 'msg': 'Fake request',
                 'command': 'secagg-delete'}

        reply2 = copy.deepcopy(reply)
        reply2["node_id"] = "party3"

        # Patch response
        self.m_requests.return_value.get_responses.return_value = Responses([reply, reply2])
        self.m_requests.return_value.send_message.return_value = 123
        self.srvkey_context._element = SecaggElementTypes.SERVER_KEY

        result = self.srvkey_context.delete(timeout=1)
        self.assertTrue(result)

        with patch("fedbiomed.researcher.secagg.SecaggContext._delete_payload") as mock_payload:
            mock_payload.return_value = ("KEY", True)
            result = self.srvkey_context.delete(timeout=1)
            self.assertTrue(result)

            mock_payload.return_value = ("KEY", False)
            result = self.srvkey_context.delete(timeout=1)
            self.assertFalse(result)

        with self.assertRaises(FedbiomedSecaggError):
            self.srvkey_context.delete(timeout="oops")


class TestSecaggBiprimeContext(BaseTestCaseSecaggContext):

    def setUp(self) -> None:
        super().setUp()

        self.mock_bpmanager.get.return_value = None
        self.biprime_context = SecaggBiprimeContext(
            parties=[environ["ID"], 'party2', 'party3']
        )

        pass

    def tearDown(self) -> None:
        super().tearDown()
        pass

    @patch('random.randrange')
    @patch("time.sleep")
    def test_biprime_context_01_payload(self, mock_time, mock_randrange):

        dummy_random = '123456'
        mock_randrange.return_value = dummy_random

        context, status = self.biprime_context._payload()
        # current test for dummy biprime generation
        self.assertDictEqual(context, {'biprime': dummy_random, 'max_keybits': 0})
        self.assertTrue(status)

    @patch('time.sleep')
    def test_biprime_context_02_secagg_delete(
            self,
            patch_time_sleep):
        """Setup then delete a secagg class"""

        reply = {'researcher_id': environ["ID"],
                 'secagg_id': self.biprime_context.secagg_id,
                 'sequence': 123,
                 'success': True,
                 'node_id': "party2",
                 'msg': 'Fake request',
                 'command': 'secagg-delete'}

        reply2 = copy.deepcopy(reply)
        reply2["node_id"] = "party3"

        # Patch response
        self.m_requests.return_value.get_responses.return_value = Responses([reply, reply2])
        self.m_requests.return_value.send_message.return_value = 123
        self.biprime_context._element = SecaggElementTypes.SERVER_KEY

        result = self.biprime_context.delete(timeout=1)
        self.assertTrue(result)

        with patch("fedbiomed.researcher.secagg.SecaggContext._delete_payload") as mock_payload:
            mock_payload.return_value = ("KEY", True)
            result = self.biprime_context.delete(timeout=1)
            self.assertTrue(result)

            mock_payload.return_value = ("KEY", False)
            result = self.biprime_context.delete(timeout=1)
            self.assertFalse(result)

        with self.assertRaises(FedbiomedSecaggError):
            self.biprime_context.delete(timeout="oops")



if __name__ == '__main__':  # pragma: no cover
    unittest.main()
