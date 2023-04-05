import unittest
from unittest.mock import patch

from testsupport.base_case import ResearcherTestCase
from testsupport.base_mocks import MockRequestMessaging

from fedbiomed.researcher.environ import environ
from fedbiomed.researcher.secagg import SecureAggregation
from fedbiomed.common.exceptions import FedbiomedSecureAggregationError, FedbiomedSecaggError


class TestSecureAggregation(MockRequestMessaging, ResearcherTestCase):

    def setUp(self) -> None:

        self.p1 = patch('fedbiomed.researcher.secagg.SecaggServkeyContext.setup')
        self.p2 = patch('fedbiomed.researcher.secagg.SecaggBiprimeContext.setup')

        self.p1.start()
        self.p2.start()

        self.secagg = SecureAggregation()

    def tearDown(self) -> None:
        self.p1.stop()
        self.p2.stop()

    def test_secure_aggregation_01_init_raises(self):
        """Tests invalid argument for __init__"""

        with self.assertRaises(FedbiomedSecureAggregationError):
            SecureAggregation(timeout="111")

        with self.assertRaises(FedbiomedSecureAggregationError):
            SecureAggregation(active="111")

        with self.assertRaises(FedbiomedSecureAggregationError):
            SecureAggregation(clipping_range="Not an integer")

        with self.assertRaises(FedbiomedSecureAggregationError):
            SecureAggregation(clipping_range=True)

    def test_secure_aggregation_02_activate(self):
        """Tests secure aggregation activation"""

        self.secagg.activate(True)
        self.assertTrue(self.secagg.active)

        self.secagg.activate(False)
        self.assertFalse(self.secagg.active)

        with self.assertRaises(FedbiomedSecureAggregationError):
            self.secagg.activate("NON-BOOL")

    def test_secure_aggregation_04_configure_round(self):
        """Test round configuration for """

        self.secagg._configure_round(
            parties=[environ["ID"], "node-1", "node-2"],
            job_id="exp-id-1",
        )

        self.assertListEqual(self.secagg.parties, [environ["ID"], "node-1", "node-2"])
        self.assertEqual(self.secagg.job_id, "exp-id-1")

        # Add new paty
        self.secagg.setup(
            parties=[environ["ID"], "node-1", "node-2", "new_party"],
            job_id="exp-id-1",
        )

        self.assertListEqual(self.secagg._biprime.parties, [environ["ID"], "node-1", "node-2", "new_party"])
        self.assertListEqual(self.secagg._servkey.parties, [environ["ID"], "node-1", "node-2", "new_party"])

        pass

    def test_secure_aggregation_05_secagg_context_ids_and_context(self):
        """Test getters for secagg biprime and servkey context ids"""
        self.secagg.setup(
            parties=[environ["ID"], "node-1", "node-2", "new_party"],
            job_id="exp-id-1",
        )

        s_id = self.secagg.servkey
        b_id = self.secagg.biprime

        self.assertTrue(s_id is not None)
        self.assertTrue(b_id is not None)

        self.assertEqual(self.secagg._biprime.secagg_id, self.secagg.biprime.secagg_id)
        self.assertEqual(self.secagg._servkey.secagg_id, self.secagg.servkey.secagg_id)

        self.secagg._biprime._context = "Test biprime"
        self.secagg._servkey._context = "Test servkey"

        s_context = self.secagg.servkey.context
        b_context = self.secagg.biprime.context

        self.assertEqual("Test biprime", b_context)
        self.assertEqual("Test servkey", s_context)

    def test_secure_aggregation_06_setup(self):
        """Test secagg setup by setting Biprime and Servkey"""

        with self.assertRaises(FedbiomedSecureAggregationError):
            self.secagg.setup(parties="oops",
                              job_id="exp-id-1")

        with self.assertRaises(FedbiomedSecureAggregationError):
            self.secagg.setup(parties=[environ["ID"], "node-1", "node-2", "new_party"],
                              job_id=1345)

        # iterate twice with same incorrect arguments
        with self.assertRaises(FedbiomedSecaggError):
            self.secagg.setup(parties=["oops"],
                              job_id="exp-id-1")
        with self.assertRaises(FedbiomedSecureAggregationError):
            self.secagg.setup(parties=["oops"],
                              job_id="exp-id-1")

        # Execute setup
        self.secagg.setup(parties=[environ["ID"], "node-1", "node-2", "new_party"],
                          job_id="exp-id-1")

    def test_secure_aggregation_07_train_arguments(self):

        self.secagg.setup(parties=[environ["ID"], "node-1", "node-2", "new_party"],
                          job_id="exp-id-1")

        args = self.secagg.train_arguments()

        self.assertListEqual(list(args.keys()), ["secagg_servkey_id", "secagg_biprime_id", "secagg_random",
                                                 "secagg_clipping_range"])

    def test_secure_aggregation_08_aggregate(self):
        """Tests aggregate method"""
        with self.assertRaises(FedbiomedSecureAggregationError):
            self.secagg.aggregate(round_=1,
                                  total_sample_size=100,
                                  model_params={'node-1': [1, 2, 3, 4, 5], 'node-2': [1, 2, 3, 4, 5]},
                                  encryption_factors={'node-1': [1], 'node-2': [1]}
                                  )

        # Configure for round
        self.secagg.setup(
            parties=[environ["ID"], "node-1", "node-2", "new_party"],
            job_id="exp-id-1",
        )

        # Raises since biprime and servkey status are False
        with self.assertRaises(FedbiomedSecureAggregationError):
            self.secagg.aggregate(round_=1,
                                  total_sample_size=100,
                                  model_params={'node-1': [1, 2, 3, 4, 5], 'node-2': [1, 2, 3, 4, 5]},
                                  encryption_factors={'node-1': [1], 'node-2': [1]}
                                  )

        # Force to set status True
        self.secagg._biprime._status = True
        self.secagg._servkey._status = True

        # Force to set context
        self.secagg._biprime._context = {'context': {'biprime': 1234}}
        self.secagg._servkey._context = {'context': {'server_key': 1234}}

        # raises error if secagg_random is set but encryption factors are not provided
        with self.assertRaises(FedbiomedSecureAggregationError):
            self.secagg.aggregate(round_=1,
                                  total_sample_size=100,
                                  model_params={'node-1': [1, 2, 3, 4, 5], 'node-2': [1, 2, 3, 4, 5]},
                                  )

        # Aggregation without secagg_random validation
        self.secagg._secagg_random = None
        agg_params = self.secagg.aggregate(round_=1,
                                           total_sample_size=100,
                                           model_params={'node-1': [1, 2, 3, 4, 5], 'node-2': [1, 2, 3, 4, 5]},
                                           encryption_factors={'node-1': [1], 'node-2': [1]}
                                           )
        self.assertTrue(len(agg_params) == 5)

        # IMPORTANT: this value has been set for biprime 1234 and servkey 1234
        # aggregation of [1], [1] will be closer to -2.8131
        self.secagg._secagg_random = -2.8131

        agg_params = self.secagg.aggregate(round_=1,
                                           total_sample_size=100,
                                           model_params={'node-1': [1, 2, 3, 4, 5], 'node-2': [1, 2, 3, 4, 5]},
                                           encryption_factors={'node-1': [1], 'node-2': [1]}
                                           )
        self.assertTrue(len(agg_params) == 5)

        # Will fail since secagg random is not correctly decrypted
        with self.assertRaises(FedbiomedSecureAggregationError):
            self.secagg._secagg_random = 2.8131
            self.secagg.aggregate(round_=1,
                                  total_sample_size=100,
                                  model_params={'node-1': [1, 2, 3, 4, 5], 'node-2': [1, 2, 3, 4, 5]},
                                  encryption_factors={'node-1': [1], 'node-2': [1]}
                                  )

    def test_secure_aggregation_09_save_state(self):
        # Configure for round
        self.secagg.setup(
            parties=[environ["ID"], "node-1", "node-2", "new_party"],
            job_id="exp-id-1",
        )

        state = self.secagg.save_state()

        self.assertEqual(state["class"], "SecureAggregation")
        self.assertEqual(state["module"], "fedbiomed.researcher.secagg._secure_aggregation")
        self.assertEqual(list(state["attributes"].keys()), ['_biprime', '_servkey', '_job_id', '_parties'])
        self.assertEqual(list(state["arguments"].keys()), ['active', 'timeout', 'clipping_range'])

        pass

    def test_secure_aggregation_10_load_state(self):
        job_id = "exp-id-1"
        parties = [environ["ID"], "node-1", "node-2", "new_party"]

        # Configure for round
        self.secagg.setup(
            parties=parties,
            job_id="exp-id-1",
        )

        biprime_id = self.secagg.biprime.secagg_id
        servkey_id = self.secagg.servkey.secagg_id

        state = self.secagg.save_state()

        # Load from state
        secagg = SecureAggregation.load_state(state)

        self.assertEqual(secagg.biprime.secagg_id, biprime_id)
        self.assertEqual(secagg.servkey.secagg_id, servkey_id)
        self.assertEqual(secagg.job_id, job_id)
        self.assertListEqual(secagg.parties, parties)

        pass


if __name__ == "__main__":
    unittest.main()
