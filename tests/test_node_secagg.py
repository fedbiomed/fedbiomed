
import unittest
import random

 #############################################################
# Import NodeTestCase before importing any FedBioMed Module
from testsupport.base_case import NodeTestCase
#############################################################

from unittest.mock import patch

import fedbiomed.node.secagg._secagg_round

from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.node.secagg._secagg_round import SecaggRound, _JLSRound, _LomRound
from fedbiomed.common.constants import SecureAggregationSchemes


class TestSecaggRound(NodeTestCase):

    def setUp(self):

        self.skmanager_p = patch.object(
            fedbiomed.node.secagg._secagg_round, "SKManager", autospec=True)  # pylint: disable=W0212
        self.bpmanager_p = patch.object(
            fedbiomed.node.secagg._secagg_round, "BPrimeManager", autospec=True)  # pylint: disable=W0212
        self.dhmanager_p = patch.object(
            fedbiomed.node.secagg._secagg_round, "DHManager", autospec=True)  # pylint: disable=W0212
        self.skmanager = self.skmanager_p.start()
        self.bpmanager = self.bpmanager_p.start()
        self.dhmanager = self.dhmanager_p.start()

        self.secagg_arguments = {
            'secagg_scheme': SecureAggregationSchemes.JOYE_LIBERT,
            'secagg_id': 'test-secagg-id',
            'secagg_clipping_range': 3,
            'secagg_random': 34,
            'parties': ['researcher-1', 'node-1', 'node-2']
        }

    def tearDown(self):
        self.skmanager_p.stop()
        self.bpmanager_p.stop()
        self.dhmanager_p.stop()

    def test_01_secagg_round_instantiation(self):
        """Tests instantiating SecaggRound"""

        self.env["FORCE_SECURE_AGGREGATION"] = True
        with self.assertRaises(FedbiomedError):
             SecaggRound(
                secagg_arguments={},
                experiment_id="test-id")

        self.env["SECURE_AGGREGATION"] = False
        with self.assertRaises(FedbiomedError):
             SecaggRound(
                secagg_arguments=self.secagg_arguments,
                experiment_id='test-id')

        self.env["SECURE_AGGREGATION"] = True
        self.secagg_arguments.pop("secagg_scheme", None)
        with self.assertRaises(FedbiomedError):
            SecaggRound(
                secagg_arguments=self.secagg_arguments,
                experiment_id="test-id")

        self.secagg_arguments["secagg_scheme"] = "opps"
        with self.assertRaises(FedbiomedError):
            SecaggRound(
                secagg_arguments=self.secagg_arguments,
                experiment_id="test-id")

    def test_02_secagg_round_jls(self):
        """Tests secagg JLSRound instantiation"""

        self.env["SECURE_AGGREGATION"] = True
        self.secagg_arguments["secagg_servkey_id"] = 'test-serv-id'
        self.secagg_arguments["secagg_biprime_id"] = 'test-bi-id'

        self.skmanager.get.return_value = {
            "parties": ['researcher-1', 'node-1', 'node-2']
        }
        self.bpmanager.get.return_value = {
            "parties": ['researcher-1', 'node-1', 'node-2']
        }

        secagg_round = SecaggRound(
            secagg_arguments=self.secagg_arguments,
            experiment_id="test-id"
        )
        self.assertIsInstance(secagg_round.scheme, _JLSRound)
        self.assertEqual(secagg_round.scheme.secagg_random, 34)

        # If biprime parties does not match ----------------------
        self.bpmanager.get.return_value = {"parties": ['researcher-12', 'node-1', 'node-2']}
        with self.assertRaises(FedbiomedError):
            SecaggRound(
                secagg_arguments=self.secagg_arguments,
                experiment_id="test-id"
            )
        # -----------------------------------------------------------

        # If skmanager parties does not match ------------------------
        self.skmanager.get.return_value = {"parties": ['researcher-12', 'node-1', 'node-2']}
        with self.assertRaises(FedbiomedError):
            SecaggRound(
                secagg_arguments=self.secagg_arguments,
                experiment_id="test-id"
            )
        # -------------------------------------------------------------

        # If skmanager is none id is wrong -------------------------------------
        self.skmanager.get.return_value = None
        with self.assertRaises(FedbiomedError):
            SecaggRound(
                secagg_arguments=self.secagg_arguments,
                experiment_id="test-id"
            )
        # ----------------------------------------------------------------------

        # If biprime id is wrong -----------------------------------------------
        self.skmanager.get.return_value = 'not-none'
        self.bpmanager.get.return_value = None
        with self.assertRaises(FedbiomedError):
            SecaggRound(
                secagg_arguments=self.secagg_arguments,
                experiment_id="test-id"
            )

        self.bpmanager.get.return_value = 'not-none'
        # ----------------------------------------------------------------------


        # If min number of parties is not respected ---------------------------
        self.secagg_arguments["parties"] = ['ops']
        with self.assertRaises(FedbiomedError):
            SecaggRound(
                secagg_arguments=self.secagg_arguments,
                experiment_id="test-id"
            )
        self.secagg_arguments["parties"] = ['researcher-1', 'node-1', 'node-2']
        # ---------------------------------------------------------------------

        # if clipping range is not valid --------------------------------------
        self.secagg_arguments["secagg_clipping_range"] = "invalid-type"
        with self.assertRaises(FedbiomedError):
            SecaggRound(
                secagg_arguments=self.secagg_arguments,
                experiment_id="test-id"
            )
        # ----------------------------------------------------------------------



    def test_03_secagg_round_jls_encrypt(self):
        """Tests JLS encrypt execution"""

        self.skmanager.get.return_value = {
            "parties": ['researcher-1', 'node-1', 'node-2'],
            "context": {"server_key": 12345}
        }
        self.bpmanager.get.return_value = {
            "parties": ['researcher-1', 'node-1', 'node-2'],
            "context": {"biprime": 1156}
        }
        secagg = SecaggRound(
            secagg_arguments=self.secagg_arguments,
            experiment_id="test-id")
        secagg.scheme.encrypt(params=[1.0, 1.0], current_round=1, weight=20)


    def test_04_secagg_round_lom_instantiate(self):
        """Tests instantiating _LomRound through secagg round"""
        self.secagg_arguments.update({
            'secagg_scheme': SecureAggregationSchemes.LOM,
            'parties': ['node-1', 'node-2']
        })
        self.dhmanager.get.return_value = {
            "parties": ['node-1', 'node-2']
        }
        secagg_round = SecaggRound(self.secagg_arguments, "test-exp-id")
        self.assertIsInstance(secagg_round.scheme, _LomRound)


        self.dhmanager.get.return_value = None
        with self.assertRaises(FedbiomedError):
            SecaggRound(self.secagg_arguments, "test-exp-id")

        self.dhmanager.get.return_value = {'parties': ['no-match']}
        with self.assertRaises(FedbiomedError):
            SecaggRound(self.secagg_arguments, "test-exp-id")

    def test_05_secagg_round_lom_encrypt(self):
        """Tests executing encrypt method of lom round"""
        self.env['ID'] = 'node-1'
        self.secagg_arguments.update({
            'secagg_scheme': SecureAggregationSchemes.LOM,
            'parties': ['node-1', 'node-2']
        })
        self.dhmanager.get.return_value = {
            "parties": ['node-1', 'node-2'],
            "context": {
                "node-2": random.randbytes(32)

            }
        }

        secagg_round = SecaggRound(self.secagg_arguments, "test-exp-id")
        result = secagg_round.scheme.encrypt(params=[1.0, 1.0], current_round=1, weight=20)
        self.assertEqual(len(result), 2)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
