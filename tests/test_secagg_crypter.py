import unittest
from math import ceil, log2
from unittest.mock import patch

from gmpy2 import mpz

from fedbiomed.common.constants import SAParameters
from fedbiomed.common.exceptions import FedbiomedSecaggCrypterError
from fedbiomed.common.secagg import EncryptedNumber, SecaggCrypter
from fedbiomed.common.secagg._jls import PublicParam


class TestSecaggCrypter(unittest.TestCase):
    biprime = 158820908809271716671659880613366104677813341255487834154303909761107215283569995523817428402987962641429395032343305343341950966867458277812575065022203120547706127493272939455658018882112230042773163870472621818892994896895819790062496734944602899772583591514631486212290112369502692304700112819186167541107

    def setUp(self) -> None:
        self.secagg_crypter = SecaggCrypter()
        pass

    def tearDown(self) -> None:
        pass

    def tests_secagg_crypter_01_setup_public_param(self):
        """Tests biprime setup"""
        pp = self.secagg_crypter._setup_public_param(biprime=12345)
        self.assertIsInstance(pp, PublicParam)
        self.assertEqual(pp.n_modulus, 12345)

    def tests_secagg_crypter_02_convert_to_encrypted_number(self):
        """Tests conversion from int to encrypted number"""
        vector = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]

        pp = self.secagg_crypter._setup_public_param(biprime=12345)

        l_encrypted_num = self.secagg_crypter._convert_to_encrypted_number(vector, pp)

        # Check each element of list is EncryptedNumber
        self.assertIsInstance(l_encrypted_num[0][0], EncryptedNumber)
        self.assertIsInstance(l_encrypted_num[2][3], EncryptedNumber)

        # Check if ciphertexts are in type of MPZ
        self.assertIsInstance(l_encrypted_num[2][3].ciphertext, mpz)
        self.assertIsInstance(l_encrypted_num[2][3].ciphertext, mpz)

        # Check ciphertext has correct value
        self.assertEqual(l_encrypted_num[0][0].ciphertext, 1)
        self.assertEqual(l_encrypted_num[2][3].ciphertext, 4)

    def test_secagg_crypter_03_apply_average(self):
        """Tests quantized divide"""

        # Test successful division

        vector = [4, 8, 12]
        result = self.secagg_crypter._apply_average(vector, 2)

        self.assertListEqual(result, [v / 2 for v in vector])

        # Test division with overflow

        vectors = [
            [-1],
            [1, 1, -1],
            [-1, 1, 1],
            [1, -1, 1],
        ]
        for vector in vectors:
            with self.assertRaises(FedbiomedSecaggCrypterError):
                self.secagg_crypter._apply_average(vector, 2)

    def test_secagg_crypter_04_apply_weighting(self):
        """Tests quantized multiply"""

        # Test successful multiply

        vector = [4, 8, 12]
        result = self.secagg_crypter._apply_weighting(vector, 2)

        self.assertListEqual(result, [v * 2 for v in vector])

        # Test multiply with overflow

        vectors = [
            [-1],
            [1, 1, -1],
            [-1, 1, 1],
            [1, -1, 1],
            [SAParameters.TARGET_RANGE],
            [2 * SAParameters.TARGET_RANGE],
            [0, SAParameters.TARGET_RANGE, 0],
            [0, 0, SAParameters.TARGET_RANGE],
        ]
        for vector in vectors:
            with self.assertRaises(FedbiomedSecaggCrypterError):
                self.secagg_crypter._apply_weighting(vector, 2)

    def test_secagg_crypter_04b_apply_weighting_target_range_bound(self):
        """_apply_weighting bound scales with target_range (FA uses a wider range)."""
        # One past the default max -> rejected with the default range...
        val = SAParameters.TARGET_RANGE
        with self.assertRaises(FedbiomedSecaggCrypterError):
            self.secagg_crypter._apply_weighting([val], 2)
        # ...but in-bounds when the wider FA range is used.
        self.assertListEqual(
            self.secagg_crypter._apply_weighting(
                [val], 2, target_range=SAParameters.FA_TARGET_RANGE
            ),
            [val * 2],
        )
        # A value at the FA bound is still rejected.
        with self.assertRaises(FedbiomedSecaggCrypterError):
            self.secagg_crypter._apply_weighting(
                [SAParameters.FA_TARGET_RANGE],
                2,
                target_range=SAParameters.FA_TARGET_RANGE,
            )

    def test_secagg_crypter_04c_encrypt_forwards_target_range(self):
        """encrypt() forwards target_range to quantize (None -> TARGET_RANGE)."""
        for given, expected in [
            (SAParameters.FA_TARGET_RANGE, SAParameters.FA_TARGET_RANGE),
            (None, SAParameters.TARGET_RANGE),  # backward-compatible default
        ]:
            with patch(
                "fedbiomed.common.secagg._secagg_crypter.quantize",
                return_value=[1, 2, 3, 4],
            ) as mock_q:
                self.secagg_crypter.encrypt(
                    num_nodes=2,
                    current_round=1,
                    params=[1.0, 2.0, 3.0, 4.0],
                    biprime=TestSecaggCrypter.biprime,
                    key=10,
                    target_range=given,
                )
            self.assertEqual(mock_q.call_args.kwargs["target_range"], expected)

    def test_secagg_crypter_04d_aggregate_forwards_target_range(self):
        """aggregate() forwards target_range to reverse_quantize."""

        def enc():
            return self.secagg_crypter.encrypt(
                num_nodes=2,
                current_round=2,
                params=[0.5, 0.8],
                biprime=TestSecaggCrypter.biprime,
                key=10,
            )

        with patch(
            "fedbiomed.common.secagg._secagg_crypter.reverse_quantize",
            return_value=[0.5, 0.8],
        ) as mock_rq:
            self.secagg_crypter.aggregate(
                current_round=2,
                num_nodes=2,
                params=[enc(), enc()],
                biprime=TestSecaggCrypter.biprime,
                key=-20,
                total_sample_size=2,
                num_expected_params=2,
                target_range=SAParameters.FA_TARGET_RANGE,
            )
        self.assertEqual(
            mock_rq.call_args.kwargs["target_range"], SAParameters.FA_TARGET_RANGE
        )

    def test_secagg_crypter_05_encrypt(self):
        """Tests encryption"""

        key = 10
        params = [10.0, 20.0, 30.0, 40.0]
        current_round = 1
        num_nodes = 2

        result = self.secagg_crypter.encrypt(
            num_nodes=num_nodes,
            current_round=current_round,
            params=params,
            biprime=TestSecaggCrypter.biprime,
            key=key,
        )

        self.assertIsInstance(result, list)

        # IMPORTANT = Bit size can change based on current_round and num_nodes
        self.assertLessEqual(
            ceil(log2(int(result[0]))), TestSecaggCrypter.biprime.bit_length() * 2
        )

        with self.assertRaises(FedbiomedSecaggCrypterError):
            result = self.secagg_crypter.encrypt(
                num_nodes=num_nodes,
                current_round=current_round,
                params="not-a-list",
                biprime=TestSecaggCrypter.biprime,
                key=key,
            )

        with self.assertRaises(FedbiomedSecaggCrypterError):
            result = self.secagg_crypter.encrypt(
                num_nodes=num_nodes,
                current_round=current_round,
                params=["not-a-float", "not-a-float"],
                biprime=TestSecaggCrypter.biprime,
                key=key,
            )
        # Not a float
        with self.assertRaises(FedbiomedSecaggCrypterError):
            result = self.secagg_crypter.encrypt(
                num_nodes=num_nodes,
                current_round=current_round,
                params=[0, 1, 2],
                biprime=TestSecaggCrypter.biprime,
                key=key,
            )

        # If JLS.aggregate raises
        with patch("fedbiomed.common.secagg._jls.UserKey"):
            # Invalid type of UserKey
            with self.assertRaises(FedbiomedSecaggCrypterError):
                result = self.secagg_crypter.encrypt(
                    num_nodes=num_nodes,
                    current_round=current_round,
                    params=params,
                    biprime=TestSecaggCrypter.biprime,
                    key=key,
                )

    def test_secagg_crypter_06_decrypt(self):
        """Tests decryption"""

        params = [0.5, 0.8, -0.5, 0.0]
        current_round = 2
        num_nodes = 2

        node_1 = self.secagg_crypter.encrypt(
            num_nodes=num_nodes,
            current_round=current_round,
            params=params,
            biprime=TestSecaggCrypter.biprime,
            key=10,
        )

        node_2 = self.secagg_crypter.encrypt(
            num_nodes=num_nodes,
            current_round=current_round,
            params=params,
            biprime=TestSecaggCrypter.biprime,
            key=10,
        )

        result = self.secagg_crypter.aggregate(
            current_round=current_round,
            num_nodes=num_nodes,
            params=[node_1, node_2],
            biprime=TestSecaggCrypter.biprime,
            key=-20,
            total_sample_size=8,
            num_expected_params=len(params),
        )

        self.assertEqual(len(result), len(params))
        self.assertTrue(
            result[0] > 0.4 or result[0] < 0.6,
            "Secure aggregation result is not closer to expected avereage",
        )

        # Test failure of JLS aggregate
        # If num of nodes does not match the number of parameters provided
        with patch(
            "fedbiomed.common.secagg._secagg_crypter.SecaggCrypter._convert_to_encrypted_number"
        ) as m:

            def side_effect(x, p):
                return x

            m.side_effect = side_effect
            with self.assertRaises(FedbiomedSecaggCrypterError):
                self.secagg_crypter.aggregate(
                    current_round=current_round,
                    num_nodes=num_nodes,
                    params=[node_1, node_2],
                    biprime=TestSecaggCrypter.biprime,
                    key=-20,
                    total_sample_size=8,
                )

        # Raise if num nodes not equal to number of parameters
        with self.assertRaises(FedbiomedSecaggCrypterError):
            params = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
            self.secagg_crypter.aggregate(
                current_round=current_round,
                num_nodes=num_nodes,
                params=params,
                biprime=TestSecaggCrypter.biprime,
                key=-20,
                total_sample_size=8,
            )

        # Invalid type of single parameter
        with self.assertRaises(FedbiomedSecaggCrypterError):
            params = [
                ["not-int", "not-int", "not-int"],
                ["not-int", "not-int", "not-int"],
            ]
            self.secagg_crypter.aggregate(
                current_round=current_round,
                num_nodes=num_nodes,
                params=params,
                key=-20,
                biprime=TestSecaggCrypter.biprime,
                total_sample_size=8,
            )

        # Unsupported list shape
        with self.assertRaises(FedbiomedSecaggCrypterError):
            params = ["not-int", "not-int"]
            self.secagg_crypter.aggregate(
                current_round=current_round,
                num_nodes=num_nodes,
                params=params,
                key=-20,
                biprime=TestSecaggCrypter.biprime,
                total_sample_size=8,
            )

    def test_secagg_crypter_07_fa_wide_range_roundtrip(self):
        """JLS round-trip with federated-analytics wide-range (~55-bit) values.

        Regression test: the Joye-Libert vector encoder packs several values into
        one plaintext. Its slot size must be derived from the (wide) FA target
        range, otherwise ~55-bit FA statistics overflow the default training-sized
        slots and corrupt the neighbouring packed value. With the encoder sized for
        FA_TARGET_RANGE the two adjacent stats (count, sum) decode independently.
        """
        C = SAParameters.FA_CLIPPING_RANGE
        T = SAParameters.FA_TARGET_RANGE
        num_nodes, current_round = 2, 1

        # 'year' [count, sum] held by each of two nodes (adjacent in the vector).
        node_1_vals = [10667.0, 21516413.0]
        node_2_vals = [17965.0, 36233008.0]

        node_1 = self.secagg_crypter.encrypt(
            num_nodes=num_nodes,
            current_round=current_round,
            params=node_1_vals,
            key=10,
            biprime=TestSecaggCrypter.biprime,
            clipping_range=C,
            weight=1,
            target_range=T,
        )
        node_2 = self.secagg_crypter.encrypt(
            num_nodes=num_nodes,
            current_round=current_round,
            params=node_2_vals,
            key=10,
            biprime=TestSecaggCrypter.biprime,
            clipping_range=C,
            weight=1,
            target_range=T,
        )

        aggregated = self.secagg_crypter.aggregate(
            current_round=current_round,
            num_nodes=num_nodes,
            params=[node_1, node_2],
            key=-20,
            biprime=TestSecaggCrypter.biprime,
            total_sample_size=num_nodes,
            clipping_range=C,
            num_expected_params=2,
            target_range=T,
        )
        # FA restores the additive sum across nodes (crypter returns the mean).
        result = [v * num_nodes for v in aggregated]

        # count = 10667 + 17965 = 28632 ; sum = 21516413 + 36233008 = 57749421
        self.assertAlmostEqual(result[0], 28632.0, delta=1.0)
        self.assertAlmostEqual(result[1], 57749421.0, delta=1.0)


if __name__ == "__main__":
    unittest.main()
