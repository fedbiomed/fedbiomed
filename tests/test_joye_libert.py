import unittest

from math import ceil, log2
from gmpy2 import mpz
from unittest.mock import patch
from fedbiomed.common.constants import VEParameters
from fedbiomed.common.secagg._jls import PublicParam, JoyeLibert, FDH, EncryptedNumber, UserKey, BaseKey, ServerKey


class TestFDH(unittest.TestCase):

    def test_fdh_01_init(self):
        """Tests wrong argument types"""

        with self.assertRaises(TypeError):
            FDH(
                bits_size=VEParameters.KEY_SIZE,
                n_modulus="1234"
            )

        with self.assertRaises(TypeError):
            FDH(
                bits_size=VEParameters.KEY_SIZE,
                n_modulus=12123
            )

        with self.assertRaises(TypeError):
            FDH(
                bits_size="not-int",
                n_modulus=12123
            )

        pass

    def test_fdh_01_H(self):
        """Tests H function"""

        n = mpz(12345)
        fdh = FDH(
            bits_size=VEParameters.KEY_SIZE,
            n_modulus=n
        )

        result_1 = fdh.H(10)
        self.assertTrue(result_1 != 0)
        self.assertIsInstance(result_1, mpz)

        result_2 = fdh.H(10)
        self.assertTrue(result_1 == result_2)


class TestPublicParam(unittest.TestCase):

    def setUp(self) -> None:
        self.n_1 = mpz(123456)
        self.n_2 = mpz(123456789)
        self.pp_1 = PublicParam(
            n_modulus=self.n_1,
            bits=VEParameters.KEY_SIZE // 2,
            hashing_function=FDH(VEParameters.KEY_SIZE, self.n_1 * self.n_1).H
        )

        self.pp_2 = PublicParam(
            n_modulus=self.n_2,
            bits=VEParameters.KEY_SIZE // 2,
            hashing_function=FDH(VEParameters.KEY_SIZE, self.n_2 * self.n_2).H
        )
        pass

    def tearDown(self) -> None:
        pass

    def test_public_param_01_equality(self):
        """Tests equality of two PublicParam"""

        self.assertFalse(self.pp_1 == self.pp_2)
        self.assertTrue(self.pp_1 == self.pp_1)

    def test_public_param_02_getters(self):
        """Tests properties"""

        self.assertEqual(self.pp_1.bits, VEParameters.KEY_SIZE // 2)
        self.assertEqual(self.pp_1.n_modulus, self.n_1)
        self.assertEqual(self.pp_1.n_square, self.n_1 * self.n_1)

    def test_public_param_03_hashing_function(self):
        """Tests hashing function defined FDH"""

        result = self.pp_1.hashing_function(10)
        self.assertTrue(result != 0)
        self.assertIsInstance(result, mpz)


class TestEncryptedNumber(unittest.TestCase):

    def setUp(self) -> None:
        n = mpz(123457)
        self.public_param = PublicParam(
            n_modulus=n,
            bits=VEParameters.KEY_SIZE // 2,
            hashing_function=FDH(VEParameters.KEY_SIZE, n * n).H
        )

        self.en = EncryptedNumber(param=self.public_param,
                                  ciphertext=10)
        pass

    def tearDown(self) -> None:
        pass

    def test_encrypted_num_01__sum_operations(self):
        """Tests normal sum operation"""
        sum_ = self.en + self.en

        # Sum should be 100
        self.assertEqual(sum_.ciphertext, 100)

        # Sum should be 10000
        sum_ = self.en + self.en + self.en + self.en
        self.assertEqual(sum_.ciphertext, 10000)

    def test_encrypted_num_02_tuple_sum(self):
        """Tests list sum"""
        ens_1 = [self.en, self.en, self.en]
        ens_2 = [self.en, self.en, self.en]

        result = [sum(ep) for ep in zip(*[ens_1, ens_2])]

        self.assertEqual(result[0].ciphertext, 100)
        self.assertEqual(result[1].ciphertext, 100)
        self.assertEqual(result[2].ciphertext, 100)

    def test_encrypted_num_03_different_public_param(self):
        """Tests error raises if PublicParams are different"""
        n = mpz(987654123)
        p_other = PublicParam(
            n_modulus=n,
            bits=VEParameters.KEY_SIZE // 2,
            hashing_function=FDH(VEParameters.KEY_SIZE, n * n).H
        )

        en = EncryptedNumber(param=p_other,
                             ciphertext=10)

        with self.assertRaises(ValueError):
            self.en + en

    def test_encrypted_num_03_sum_different_types(self):
        """Test if raises error when EN summed with int"""
        with self.assertRaises(TypeError):
            self.en.__repr__
            self.en + 15


class TestBaseKey(unittest.TestCase):

    def setUp(self) -> None:
        self.n = mpz(123457)
        self.key = 191919191919191
        self.public_param = PublicParam(
            n_modulus=self.n,
            bits=VEParameters.KEY_SIZE // 2,
            hashing_function=FDH(VEParameters.KEY_SIZE, self.n * self.n).H
        )

        self.base_key = BaseKey(public_param=self.public_param,
                                key=self.key)

    def test_base_key_01_public_param(self):
        """Check public param property and repr"""
        self.assertEqual(self.public_param, self.base_key.public_param)
        self.assertTrue(self.base_key.__repr__ != '')

    def test_base_key_02_hash(self):
        """Tests hash magic method"""
        result = hash(self.base_key)
        self.assertEqual(result, self.key)

    def test_base_key_03_equal(self):
        self.assertTrue(self.base_key == self.base_key)

        # Create another base key
        n = mpz(123457)
        key = 19191919191919121
        self.public_param = PublicParam(
            n_modulus=n,
            bits=VEParameters.KEY_SIZE // 2,
            hashing_function=FDH(VEParameters.KEY_SIZE, n * n).H
        )

        base_key_2 = BaseKey(public_param=self.public_param,
                             key=key)

        self.assertFalse(self.base_key == base_key_2)

        # Test compare different type
        with self.assertRaises(TypeError):
            self.base_key == 10

    def test_base_key_03_populate_tau(self):
        result = self.base_key._populate_tau(tau=1, len_=10)
        self.assertEqual(len(result), 10)
        self.assertTrue(all([r.bit_length() <= VEParameters.KEY_SIZE // 8 for r in result]))


class TestUserKey(unittest.TestCase):

    def setUp(self) -> None:
        self.n = mpz(123457)
        self.key = 191919191919191
        self.public_param = PublicParam(
            n_modulus=self.n,
            bits=VEParameters.KEY_SIZE // 2,
            hashing_function=FDH(VEParameters.KEY_SIZE, self.n * self.n).H
        )

        self.user_key = UserKey(public_param=self.public_param,
                                key=self.key)

    def test_user_key_01_equality(self):
        server_key = ServerKey(public_param=self.public_param,
                               key=self.key)

        # Can not compare user key and server key
        with self.assertRaises(TypeError):
            self.user_key == server_key

    def test_user_key_02_encrypt(self):
        """Tests UserKey encryption"""

        plaintext = [mpz(10), mpz(10), mpz(10)]
        en = self.user_key.encrypt(plaintext=plaintext,
                                   tau=1)

        self.assertIsInstance(en, list)
        self.assertEqual(len(en), 3)

        with self.assertRaises(TypeError):
            self.user_key.encrypt(plaintext="not-a-list",
                                  tau=1)


class TestServerKey(unittest.TestCase):

    def test_server_key_decryption(self):
        n = mpz(123457)
        public_param = PublicParam(
            n_modulus=n,
            bits=VEParameters.KEY_SIZE // 2,
            hashing_function=FDH(VEParameters.KEY_SIZE, n * n).H
        )

        user_key_1 = UserKey(public_param=public_param,
                             key=10)

        user_key_2 = UserKey(public_param=public_param,
                             key=10)

        server_key = ServerKey(public_param=public_param,
                               key=-20)

        plaintext = [mpz(10), mpz(10), mpz(10)]
        en_1 = user_key_1.encrypt(plaintext=plaintext,
                                  tau=1)
        en_2 = user_key_2.encrypt(plaintext=plaintext,
                                  tau=1)
        en_1 = [EncryptedNumber(public_param, en) for en in en_1]
        en_2 = [EncryptedNumber(public_param, en) for en in en_2]

        sum_ = [sum(en) for en in zip(*[en_1, en_2])]

        dec = server_key.decrypt(sum_, tau=1)
        self.assertEqual(dec, [20, 20, 20])


class TestJoyeLibert(unittest.TestCase):

    def setUp(self) -> None:
        self.jl = JoyeLibert()

        p = mpz(
            7801876574383880214548650574033350741129913580793719706746361606042541080141291132224899113047934760791108387050756752894517232516965892712015132079112571
        )
        q = mpz(
            7755946847853454424709929267431997195175500554762787715247111385596652741022399320865688002114973453057088521173384791077635017567166681500095602864712097
        )

        n = p * q
        self.public_param = PublicParam(
            n_modulus=n,
            bits=VEParameters.KEY_SIZE // 2,
            hashing_function=FDH(VEParameters.KEY_SIZE, n * n).H
        )

        self.user_key_1 = UserKey(public_param=self.public_param, key=10)
        self.user_key_2 = UserKey(public_param=self.public_param, key=10)
        self.server_key = ServerKey(public_param=self.public_param, key=-20)

    def test_joye_libert_01_protect(self):
        """Tests protect method """
        plaintext = [10, 10, 10]

        en = self.jl.protect(public_param=self.public_param,
                             user_key=self.user_key_1,
                             tau=1,
                             x_u_tau=plaintext,
                             n_users=2)

        # Expected length is 1 due to vector encoder
        self.assertTrue(len(en) == 1)

        plaintext = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                     11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                     11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                     11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                     11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                     ]

        en = self.jl.protect(public_param=self.public_param,
                             user_key=self.user_key_1,
                             tau=1,
                             x_u_tau=plaintext,
                             n_users=2)

        # Expected length is 2 due to vector encoder
        self.assertEqual(len(en), 2)

        with self.assertRaises(TypeError):
            self.jl.protect(public_param=self.public_param,
                            user_key='in-valid-user-key',
                            tau=1,
                            x_u_tau=plaintext,
                            n_users=2)

        with self.assertRaises(ValueError):
            self.jl.protect(public_param=PublicParam(n_modulus=mpz(1111),
                                                     bits=VEParameters.KEY_SIZE // 2,
                                                     hashing_function=
                                                     FDH(VEParameters.KEY_SIZE, mpz(1111) * mpz(1111)).H),
                            user_key=self.user_key_2,
                            tau=1,
                            x_u_tau=plaintext,
                            n_users=2)

        with self.assertRaises(TypeError):
            self.jl.protect(public_param=self.public_param,
                            user_key=self.user_key_1,
                            tau=1,
                            x_u_tau='invalid-plaintext',
                            n_users=2)

    def test_joye_libert_02_aggregate(self):

        plaintext = [10, 10, 10]

        en_1 = self.jl.protect(public_param=self.public_param,
                               user_key=self.user_key_1,
                               tau=1,
                               x_u_tau=plaintext,
                               n_users=2)

        en_2 = self.jl.protect(public_param=self.public_param,
                               user_key=self.user_key_2,
                               tau=1,
                               x_u_tau=plaintext,
                               n_users=2)

        en_1_ = [EncryptedNumber(self.public_param, int(en)) for en in en_1]
        en_2_ = [EncryptedNumber(self.public_param, int(en)) for en in en_2]

        agg = self.jl.aggregate(sk_0=self.server_key,
                                tau=1,
                                list_y_u_tau=[en_1_, en_2_])

        self.assertListEqual(agg, [20, 20, 20])