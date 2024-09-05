import re
from secrets import token_bytes
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from requests import Request

from fedbiomed.common.constants import SecureAggregationSchemes
from fedbiomed.common.secagg import LOM
from fedbiomed.common.utils import quantize, multiply
from testsupport.base_case import ResearcherTestCase
from testsupport.base_mocks import MockRequestGrpc, MockRequestModule

from fedbiomed.researcher.environ import environ
from fedbiomed.researcher.secagg import SecureAggregation, JoyeLibertSecureAggregation, LomSecureAggregation
from fedbiomed.common.exceptions import FedbiomedSecureAggregationError, FedbiomedSecaggCrypterError


class TestJLSecureAggregation(MockRequestModule, ResearcherTestCase):

    def setUp(self) -> None:

        super().setUp(module="fedbiomed.researcher.secagg._secagg_context.Requests")
        self.p1 = patch('fedbiomed.researcher.secagg._secure_aggregation.SecaggServkeyContext.setup', autospec=True)
        self.p2 = patch('fedbiomed.researcher.secagg._secure_aggregation.SecaggBiprimeContext.setup', autospec=True)

        self.p1.start()
        self.p2.start()

        self.secagg = JoyeLibertSecureAggregation()

    def tearDown(self) -> None:
        super().tearDown()
        self.p1.stop()
        self.p2.stop()

    def test_jl_secure_aggregation_01_init_raises(self):
        """Tests invalid argument for __init__"""

        with self.assertRaises(FedbiomedSecureAggregationError):
            JoyeLibertSecureAggregation(active="111")

        with self.assertRaises(FedbiomedSecureAggregationError):
            JoyeLibertSecureAggregation(clipping_range="Not an integer")

        with self.assertRaises(FedbiomedSecureAggregationError):
            JoyeLibertSecureAggregation(clipping_range=True)

    def test_jl_secure_aggregation_02_activate(self):
        """Tests secure aggregation activation"""

        self.secagg.activate(True)
        self.assertTrue(self.secagg.active)

        self.secagg.activate(False)
        self.assertFalse(self.secagg.active)

        with self.assertRaises(FedbiomedSecureAggregationError):
            self.secagg.activate("NON-BOOL")

    def test_jl_secure_aggregation_04_configure_round(self):
        """Test round configuration for """

        self.secagg._configure_round(
            parties=[environ["ID"], "node-1", "node-2"],
            experiment_id="exp-id-1",
        )

        self.assertListEqual(self.secagg.parties, [environ["ID"], "node-1", "node-2"])
        self.assertEqual(self.secagg.experiment_id, "exp-id-1")

        # Add new paty
        self.secagg.setup(
            parties=[environ["ID"], "node-1", "node-2", "new_party"],
            experiment_id="exp-id-1",
        )

        self.assertListEqual(self.secagg._parties, [environ["ID"], "node-1", "node-2", "new_party"])
        self.assertListEqual(self.secagg._parties, [environ["ID"], "node-1", "node-2", "new_party"])

        self.assertIsNotNone(self.secagg._biprime)
        self.assertIsNotNone(self.secagg._servkey)

    def test_jl_secure_aggregation_05_secagg_context_ids_and_context(self):
        """Test getters for secagg biprime and servkey context ids"""
        self.secagg.setup(
            parties=[environ["ID"], "node-1", "node-2", "new_party"],
            experiment_id="exp-id-1",
        )

        s_id = self.secagg.servkey
        b_id = self.secagg.biprime

        self.assertTrue(s_id is not None)
        self.assertTrue(b_id is not None)

    def test_jl_secure_aggregation_06_setup(self):
        """Test secagg setup by setting Biprime and Servkey"""

        with self.assertRaises(FedbiomedSecureAggregationError):
            self.secagg.setup(parties="oops",
                              experiment_id="exp-id-1")

        with self.assertRaises(FedbiomedSecureAggregationError):
            self.secagg.setup(parties=[environ["ID"], "node-1", "node-2", "new_party"],
                              experiment_id=1345)

        # Execute setup
        self.secagg.setup(parties=[environ["ID"], "node-1", "node-2", "new_party"],
                          experiment_id="exp-id-1")

    def test_jl_secure_aggregation_07_train_arguments(self):

        self.secagg.setup(parties=[environ["ID"], "node-1", "node-2", "new_party"],
                          experiment_id="exp-id-1")

        args = self.secagg.train_arguments()
        self.assertListEqual(list(args.keys()), ['secagg_random', 'secagg_clipping_range', 'secagg_scheme',
                                                 'parties', 'secagg_servkey_id', 'secagg_biprime_id'])

    def test_jl_secure_aggregation_08_aggregate(self):
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
            experiment_id="exp-id-1",
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
        with self.assertRaises(FedbiomedSecaggCrypterError):
            self.secagg.aggregate(round_=1,
                                  total_sample_size=100,
                                  model_params={'node-1': [1, 2, 3, 4, 5], 'node-2': [1, 2, 3, 4, 5]},
                                  )

        with self.assertRaises(FedbiomedSecureAggregationError):
            agg_params = self.secagg.aggregate(
                round_=1,
                total_sample_size=100,
                model_params={'node-1': [1, 2, 3, 4, 5], 'node-2': [1, 2, 3, 4, 5]},
                encryption_factors={'node-1': None, 'node-2': [1]},
                num_expected_params=5)


        # Aggregation without secagg_random validation
        self.secagg._secagg_random = None
        agg_params = self.secagg.aggregate(
            round_=1,
            total_sample_size=100,
            model_params={'node-1': [1, 2, 3, 4, 5], 'node-2': [1, 2, 3, 4, 5]},
            encryption_factors={'node-1': [1], 'node-2': [1]},
            num_expected_params=5)
        self.assertTrue(len(agg_params) == 5)

        # IMPORTANT: this value has been set for biprime 1234 and servkey 1234
        # aggregation of [1], [1] will be closer to -2.9988
        self.secagg._secagg_random = -2.9988
        agg_params = self.secagg.aggregate(
            round_=1,
            total_sample_size=100,
            model_params={'node-1': [1, 2, 3, 4, 5], 'node-2': [1, 2, 3, 4, 5]},
            encryption_factors={'node-1': [1], 'node-2': [1]},
            num_expected_params=5)
        self.assertTrue(len(agg_params) == 5)

        # Will fail since secagg random is not correctly decrypted
        with self.assertRaises(FedbiomedSecureAggregationError):
            self.secagg._secagg_random = 2.9988
            self.secagg.aggregate(round_=1,
                                  total_sample_size=100,
                                  model_params={'node-1': [1, 2, 3, 4, 5], 'node-2': [1, 2, 3, 4, 5]},
                                  encryption_factors={'node-1': [1], 'node-2': [1]}
                                  )

    def test_jl_secure_aggregation_09_save_state_breakpoint(self):
        # Configure for round
        self.secagg.setup(
            parties=[environ["ID"], "node-1", "node-2", "new_party"],
            experiment_id="exp-id-1",
        )

        state = self.secagg.save_state_breakpoint()

        self.assertEqual(state["class"], "JoyeLibertSecureAggregation")
        self.assertEqual(state["module"], "fedbiomed.researcher.secagg._secure_aggregation")
        self.assertEqual(list(state["attributes"].keys()), ['_experiment_id', '_parties', '_biprime', '_servkey'])
        self.assertEqual(list(state["arguments"].keys()), ['active', 'clipping_range'])

    def test_jl_secure_aggregation_10_load_state_breakpoint(self):
        experiment_id = "exp-id-1"
        parties = [environ["ID"], "node-1", "node-2", "new_party"]

        # Configure for round
        self.secagg.setup(
            parties=parties,
            experiment_id="exp-id-1",
        )

        biprime_id = self.secagg.biprime.secagg_id
        servkey_id = self.secagg.servkey.secagg_id

        state = self.secagg.save_state_breakpoint()

        # Load from state
        secagg = JoyeLibertSecureAggregation.load_state_breakpoint(state)

        self.assertEqual(secagg.biprime.secagg_id, biprime_id)
        self.assertEqual(secagg.servkey.secagg_id, servkey_id)
        self.assertEqual(secagg.experiment_id, experiment_id)
        self.assertListEqual(secagg.parties, parties)


class TestSecureAggregationWrapper( ResearcherTestCase):
    def check_specific_method_belongs_to_class(self, original_obj, obj):
        methods_names = dir(original_obj)

        for method in methods_names:
            if not re.compile('^_').match(method):
                if not hasattr(obj, method):
                    self.fail(f"method {method} doesnot belong to object {obj}")

    def test_secure_aggregation_01_init(self):
        sa = SecureAggregation(scheme=SecureAggregationSchemes.JOYE_LIBERT)
        self.check_specific_method_belongs_to_class(JoyeLibertSecureAggregation, sa)

        sa = SecureAggregation(scheme=SecureAggregationSchemes.LOM)
        self.check_specific_method_belongs_to_class(LomSecureAggregation, sa)

        # defaults to LOM secure aggregation
        sa = SecureAggregation(scheme=SecureAggregationSchemes.NONE)
        self.check_specific_method_belongs_to_class(LomSecureAggregation, sa)

    def test_secure_aggregation_02_save_and_load_breakpoint(self):
        """Tests secure aggregation load and save breakpoints"""
        for scheme, cl in zip(SecureAggregationSchemes, (LomSecureAggregation, JoyeLibertSecureAggregation, LomSecureAggregation)):
            sa = SecureAggregation(scheme=scheme)
            state = sa.__getattr__("save_state_breakpoint")()
            self.assertDictContainsSubset({"attributes": {},
                                           "class": 'SecureAggregation',
                                           "module": "fedbiomed.researcher.secagg._secure_aggregation",
                                           "arguments": {"scheme": scheme.value}},
                                          state
                                          )
            eval(f'exec("from {state["module"]} import {state["class"]}")')
            loaded_sa = SecureAggregation.load_state_breakpoint(state)
            self.check_specific_method_belongs_to_class(loaded_sa, cl)

class TestLomSecureAggregation(MockRequestModule, ResearcherTestCase):

    def setUp(self) -> None:
        super().setUp(module="fedbiomed.researcher.secagg._secagg_context.Requests")
        self.clipping_range = 10**3
        self.args = {'active': True,
                     'clipping_range': self.clipping_range}
        self.lom = LomSecureAggregation(**self.args)
        self.parties = [environ["ID"], "node-1", "node-2", "node-3"]
        self.experiment_id = 'my_experiment_id'


    def create_protected_vector(self, round, params_1, params_2, params_3, clipping_range, weight):
        p_secrets_1 = {'node-2': b'\x02' * 32, 'node-3': b'\x02' * 32}
        p_secrets_2 = {'node-1': b'\x02' * 32, 'node-3': b'\x02' * 32}
        p_secrets_3 = {'node-1': b'\x02' * 32, 'node-2': b'\x02' * 32}
        nonce = token_bytes(16)
        lom_1 = LOM(nonce=nonce)
        lom_2 = LOM(nonce=nonce)
        lom_3 = LOM(nonce=nonce)

        (params_1, params_2, params_3) = (quantize(params_1, clipping_range),
                                          quantize(params_2, clipping_range),
                                          quantize(params_3, clipping_range))

        params_1, params_2, params_3 = multiply(params_1, weight), multiply(params_2, weight), multiply(params_3, weight)
        pv = []
        for n, l, s, p in zip(self.parties[1:],
                              (lom_1, lom_2, lom_3,),
                              (p_secrets_1, p_secrets_2, p_secrets_3,),
                              (params_1, params_2, params_3,)):
            pv.append(l.protect(n, s, round, p, self.parties[1:]))
        return pv

    def test_lom_secagg_01_train_arg(self):
        """Test train arguments and dh context"""
        t_arg = self.lom.train_arguments()
        self.assertDictEqual(t_arg, {'secagg_random': None,
                                     'secagg_clipping_range': self.clipping_range,
                                     'secagg_scheme': SecureAggregationSchemes.LOM.value,
                                     'parties': None,
                                     'secagg_dh_id': None})

    def test_lom_secagg_02_setup(self):
        # testing case where all nodes have replied (status = True)
        fake_replies = MagicMock(return_value={n: MagicMock(spec=Request, node_id = n, success = True) for n in self.parties[1:]})
        send_fed_req = {'replies': fake_replies,
                        'errors': MagicMock(return_value={})}
        super().setUp(module="fedbiomed.researcher.secagg._secagg_context.Requests",
                      send_fed_req_conf=send_fed_req)
        status = self.lom.setup(self.parties, self.experiment_id)
        self.assertIsNotNone(self.lom.dh)
        self.assertTrue(status)
        self.assertIsInstance(status, bool)

        # testing case where last node is faulty (last one)
        # 1. success is False
        fake_replies.return_value["node-3"] = MagicMock(spec=Request, node_id = "node-3", success = False)
        send_fed_req = {'replies': fake_replies,
                        'errors': MagicMock(return_value={})}
        super().setUp(module="fedbiomed.researcher.secagg._secagg_context.Requests",
                      send_fed_req_conf=send_fed_req)
        self.lom = LomSecureAggregation(**self.args)
        status = self.lom.setup(self.parties, self.experiment_id)
        self.assertFalse(status)
        self.assertIsInstance(status, bool)

        state = self.lom.save_state_breakpoint()
        lom = LomSecureAggregation.load_state_breakpoint(state)

    def test_lom_secagg_03_aggregate(self):
        fake_replies = MagicMock(return_value={n: MagicMock(spec=Request, node_id = n, success = True) for n in self.parties[1:]})
        send_fed_req = {'replies': fake_replies,
                        'errors': MagicMock(return_value={})}
        super().setUp(module="fedbiomed.researcher.secagg._secagg_context.Requests",
                      send_fed_req_conf=send_fed_req)
        self.lom.setup(self.parties, self.experiment_id)
        self.lom._secagg_random = None

        pv = self.create_protected_vector(1, [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], self.clipping_range, 5)
        agg_params = self.lom.aggregate(
                round_=1,
                total_sample_size=5,
                model_params={n: pv[i] for i, n in enumerate(self.parties[1:])},
                encryption_factors={'node-1': [5555], 'node-2': [5555], 'node-3': [5555]})


        expected_agg_params = np.sum([[1, 2, 3, 4, 5],
                                      [1, 2, 3, 4, 5],
                                      [1, 2, 3, 4, 5]], axis=0) + 2 * self.clipping_range
        for ouput_val, expected_val in zip(agg_params, expected_agg_params):
            self.assertTrue(np.isclose(np.round(ouput_val), expected_val))
    def test_lom_secagg_04_aggregate_errors(self):
        ######## WARNING #######
        # NEVER touch  self.lom._dh
        # FOR UNKNOWN REASON, IT CONFLICTS WITH OTHER TESTS AND EVERYTHING WILL FAIL
        # MOCKED RESEATCHER_ID WILL NOT BE THE SAME AS environ["ID"], WHICH IS ASSUMED
        # FOR OTHER TESTS
        with self.assertRaises(FedbiomedSecureAggregationError):
            self.lom.aggregate(
                round_=1,
                total_sample_size=100,
                model_params={'node-1': [1, 2, 3, 4, 5], 'node-2': [1, 2, 3, 4, 5]},
                encryption_factors={'node-1': [1], 'node-2': [1]},
                num_expected_params=5)

        fake_replies = MagicMock(return_value={n: MagicMock(spec=Request, node_id = n, success = True) for n in self.parties[1:]})
        fake_replies.return_value["node-3"] = MagicMock(spec=Request, node_id = "node-3", success = False)
        send_fed_req = {'replies': fake_replies,
                        'errors': MagicMock(return_value={})}
        super().setUp(module="fedbiomed.researcher.secagg._secagg_context.Requests",
                      send_fed_req_conf=send_fed_req)
        with self.assertRaises(FedbiomedSecureAggregationError):
            lom = LomSecureAggregation(**self.args)
            lom.setup(self.parties, self.experiment_id)
            lom.aggregate(
                    round_=1,
                    total_sample_size=100,
                    model_params={'node-1': [1, 2, 3, 4, 5], 'node-2': [1, 2, 3, 4, 5]},
                    encryption_factors={'node-1': [1], 'node-2': [1]},
                    num_expected_params=5)

        # raise error in `_validate` (aggregation has failed due to incorrect encryption)
        with self.assertRaises(FedbiomedSecureAggregationError):
            fake_replies = MagicMock(return_value={n: MagicMock(spec=Request, node_id = n, success = True) for n in self.parties[1:]})
            send_fed_req = {'replies': fake_replies,
                            'errors': MagicMock(return_value={})}
            super().setUp(module="fedbiomed.researcher.secagg._secagg_context.Requests",
                          send_fed_req_conf=send_fed_req)
            lom = LomSecureAggregation(**self.args)
            lom.setup(self.parties, self.experiment_id)
            lom.aggregate(
                    round_=1,
                    total_sample_size=100,
                    model_params={'node-1': [1, 2, 3, 4, 5], 'node-2': [1, 2, 3, 4, 5]},
                    encryption_factors={'node-1': [1], 'node-2': [1]},
                    num_expected_params=5)


if __name__ == "__main__":
    unittest.main()
