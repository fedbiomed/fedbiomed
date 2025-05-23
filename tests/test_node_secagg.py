import unittest
import os
import tempfile
import random


from unittest.mock import patch

import fedbiomed.node.secagg._secagg_round

from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.node.secagg._secagg_round import SecaggRound, _JLSRound, _LomRound
from fedbiomed.common.constants import SecureAggregationSchemes


class TestSecaggRound(unittest.TestCase):
    def setUp(self):
        self.skmanager_p = patch(
            "fedbiomed.node.secagg._secagg_round.SecaggServkeyManager", autospec=True
        )  # pylint: disable=W0212
        self.dhmanager_p = patch(
            "fedbiomed.node.secagg._secagg_round.SecaggDhManager", autospec=True
        )  # pylint: disable=W0212

        self.skmanager = self.skmanager_p.start()
        self.dhmanager = self.dhmanager_p.start()

        self.temp_dir = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.temp_dir.name, "test.json")

        self.secagg_arguments = {
            "secagg_scheme": SecureAggregationSchemes.JOYE_LIBERT,
            "secagg_id": "test-secagg-id",
            "secagg_clipping_range": 3,
            "secagg_random": 34,
            "parties": ["researcher-1", "node-1", "node-2"],
        }

    def tearDown(self):
        self.temp_dir.cleanup()
        self.skmanager_p.stop()
        self.dhmanager_p.stop()

    def test_01_secagg_round_instantiation(self):
        """Tests instantiating SecaggRound"""

        with self.assertRaises(FedbiomedError):
            SecaggRound(
                db=self.db,
                node_id="test-node-id",
                force_secagg=True,
                secagg_active=True,
                secagg_arguments={},
                experiment_id="test-id",
            )

        with self.assertRaises(FedbiomedError):
            SecaggRound(
                db=self.db,
                node_id="test-node-id",
                secagg_active=False,
                force_secagg=False,
                secagg_arguments=self.secagg_arguments,
                experiment_id="test-id",
            )

        self.secagg_arguments.pop("secagg_scheme", None)
        with self.assertRaises(FedbiomedError):
            SecaggRound(
                db=self.db,
                node_id="test-node-id",
                secagg_active=True,
                force_secagg=True,
                secagg_arguments=self.secagg_arguments,
                experiment_id="test-id",
            )

        self.secagg_arguments["secagg_scheme"] = "opps"
        with self.assertRaises(FedbiomedError):
            SecaggRound(
                db=self.db,
                node_id="test-node-id",
                secagg_active=True,
                force_secagg=True,
                secagg_arguments=self.secagg_arguments,
                experiment_id="test-id",
            )

    def test_02_secagg_round_jls(self):
        """Tests secagg JLSRound instantiation"""

        self.secagg_arguments["secagg_servkey_id"] = "test-serv-id"

        self.skmanager.return_value.get.return_value = {
            "parties": ["researcher-1", "node-1", "node-2"]
        }
        secagg_round = SecaggRound(
            db=self.db,
            node_id="test-node-id",
            secagg_active=True,
            force_secagg=True,
            secagg_arguments=self.secagg_arguments,
            experiment_id="test-id",
        )
        self.assertIsInstance(secagg_round.scheme, _JLSRound)
        self.assertEqual(secagg_round.scheme.secagg_random, 34)

        # If skmanager parties does not match ------------------------
        self.skmanager.return_value.get.return_value = {
            "parties": ["researcher-12", "node-1", "node-2"]
        }
        with self.assertRaises(FedbiomedError):
            SecaggRound(
                db=self.db,
                node_id="test-node-id",
                secagg_active=True,
                force_secagg=True,
                secagg_arguments=self.secagg_arguments,
                experiment_id="test-id",
            )
        # -------------------------------------------------------------

        # If skmanager is none id is wrong -------------------------------------
        self.skmanager.return_value.get.return_value = None
        with self.assertRaises(FedbiomedError):
            SecaggRound(
                db=self.db,
                node_id="test-node-id",
                secagg_active=True,
                force_secagg=True,
                secagg_arguments=self.secagg_arguments,
                experiment_id="test-id",
            )
        # ----------------------------------------------------------------------

        # If min number of parties is not respected ---------------------------
        self.secagg_arguments["parties"] = ["ops"]
        with self.assertRaises(FedbiomedError):
            SecaggRound(
                db=self.db,
                node_id="test-node-id",
                secagg_active=True,
                force_secagg=True,
                secagg_arguments=self.secagg_arguments,
                experiment_id="test-id",
            )
        self.secagg_arguments["parties"] = ["researcher-1", "node-1", "node-2"]
        # ---------------------------------------------------------------------

        # if clipping range is not valid --------------------------------------
        self.secagg_arguments["secagg_clipping_range"] = "invalid-type"
        with self.assertRaises(FedbiomedError):
            SecaggRound(
                db=self.db,
                node_id="test-node-id",
                secagg_active=True,
                force_secagg=True,
                secagg_arguments=self.secagg_arguments,
                experiment_id="test-id",
            )
        # ----------------------------------------------------------------------

    def test_03_secagg_round_jls_encrypt(self):
        """Tests JLS encrypt execution"""

        self.skmanager.return_value.get.return_value = {
            "parties": ["researcher-1", "node-1", "node-2"],
            "context": {"server_key": 12345, "biprime": 1156},
        }
        secagg = SecaggRound(
            db=self.db,
            node_id="test-node-id",
            secagg_active=True,
            force_secagg=True,
            secagg_arguments=self.secagg_arguments,
            experiment_id="test-id",
        )
        secagg.scheme.encrypt(params=[1.0, 1.0], current_round=1, weight=20)

    def test_04_secagg_round_lom_instantiate(self):
        """Tests instantiating _LomRound through secagg round"""
        self.secagg_arguments.update(
            {
                "secagg_scheme": SecureAggregationSchemes.LOM,
                "parties": ["node-1", "node-2"],
            }
        )
        self.dhmanager.return_value.get.return_value = {"parties": ["node-1", "node-2"]}
        secagg_round = SecaggRound(
            db=self.db,
            node_id="test-node-id",
            secagg_active=True,
            force_secagg=True,
            secagg_arguments=self.secagg_arguments,
            experiment_id="test-exp-id",
        )
        self.assertIsInstance(secagg_round.scheme, _LomRound)

        self.dhmanager.return_value.get.return_value = None
        with self.assertRaises(FedbiomedError):
            SecaggRound(
                db=self.db,
                node_id="test-node-id",
                secagg_active=True,
                force_secagg=True,
                secagg_arguments=self.secagg_arguments,
                experiment_id="test-exp-id",
            )

        self.dhmanager.return_value.get.return_value = {"parties": ["no-match"]}
        with self.assertRaises(FedbiomedError):
            SecaggRound(
                db=self.db,
                node_id="test-node-id",
                secagg_active=True,
                force_secagg=True,
                secagg_arguments=self.secagg_arguments,
                experiment_id="test-exp-id",
            )

    def test_05_secagg_round_lom_encrypt(self):
        """Tests executing encrypt method of lom round"""
        self.secagg_arguments.update(
            {
                "secagg_scheme": SecureAggregationSchemes.LOM,
                "parties": ["node-1", "node-2"],
            }
        )
        self.dhmanager.return_value.get.return_value = {
            "parties": ["node-1", "node-2"],
            "context": {"node-2": random.randbytes(32)},
        }

        secagg_round = SecaggRound(
            db=self.db,
            node_id="node-1",
            secagg_active=True,
            force_secagg=True,
            secagg_arguments=self.secagg_arguments,
            experiment_id="test-exp-id",
        )
        result = secagg_round.scheme.encrypt(
            params=[1.0, 1.0], current_round=1, weight=20
        )
        self.assertEqual(len(result), 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
