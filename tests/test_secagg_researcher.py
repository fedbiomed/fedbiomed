import unittest
from typing import Union
from unittest.mock import patch

import testsupport.mock_researcher_environ  ## noqa (remove flake8 false warning)
from testsupport.fake_responses import FakeResponses

from fedbiomed.common.exceptions import FedbiomedSecaggError
from fedbiomed.researcher.environ import environ
from fedbiomed.researcher.secagg import SecaggServkeyContext, SecaggBiprimeContext, SecaggContext

class TestSecaggResearcher(unittest.TestCase):
    """ Test for researcher's secagg module"""

    @classmethod
    def setUpClass(cls) -> None:
        pass

    @classmethod
    def tearDownClass(cls) -> None:
        pass

    def setUp(self):
        # Define patchers that are not modified during the test
        self.patchers = [
            patch('fedbiomed.researcher.requests.Requests.__init__',
                  return_value=None),
        ]
        # Note: we could mock Validate, but we considered OK to use the real class

        for patcher in self.patchers:
            patcher.start()

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
        for parties in parties_set:
            # test
            servkey = SecaggServkeyContext(parties)
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
                SecaggServkeyContext(parties)
            with self.assertRaises(FedbiomedSecaggError):
                SecaggBiprimeContext(parties)            

    def test_secagg_02_dummy_abstract(self):
        """Dummy test for abstract methods"""

        # Want a dummy test for abstract method (unused) code to keep code coverage
        patcher = patch.multiple(SecaggContext, __abstractmethods__=set())
        patcher.start()
        dummy = SecaggContext(['un', 'deux', 'trois'])
        dummy._payload()
        patcher.stop()

        # no check, just for coverage

    @patch('fedbiomed.researcher.requests.Requests.send_message')
    @patch('fedbiomed.researcher.requests.Requests.get_responses')
    def test_secagg_03_setup_ok(
            self,
            patch_requests_get_responses,
            patch_requests_send_message):
        """Correctly setup a secagg class"""

        # prepare
        parties = ['party1', 'party2', 'party3']

        class FakeRequests():
            def __init__(self):
                self.messages = []
                self.sequence = 0

            def send_message_side_effect(self, msg: dict, client: str = None, add_sequence: bool = False) -> \
                    Union[int, None]:
                self.messages.append([client, self.sequence, msg])
                self.sequence += 1

            def get_responses_side_effect(
                    self,
                    look_for_commands: list,
                    timeout: float = None,
                    only_successful: bool = True,
                    while_responses: bool = True) -> FakeResponses:
                import time ; time.sleep(1)
                print(len(self.messages))
                print(self.messages)
                return FakeResponses([])

        fake_requests = FakeRequests()
        patch_requests_send_message.side_effect = fake_requests.send_message_side_effect
        patch_requests_get_responses.side_effect = fake_requests.get_responses_side_effect

        context = SecaggServkeyContext(parties)
        context.setup()





if __name__ == '__main__':  # pragma: no cover
    unittest.main()
