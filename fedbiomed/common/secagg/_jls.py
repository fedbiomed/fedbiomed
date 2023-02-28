"""
### **Joye-Libert secure aggregation scheme (JL) and its threshold-variant (TJL)**

This module contains a python implementation of the Joye-Libert scheme and the threshold-variant of Joye-Libert scheme. The original scheme of Joye-Libert can be found here [1]. The threshold variant is defined here [2].

*Implemented by: Mohamad Mansouri (mohamad.mansouri@thalesgroup.com)*

[1] *Marc Joye and Benoît Libert. A scalable scheme for
privacy-preserving aggregation of time-series data. In
Ahmad-Reza Sadeghi, editor, Financial Cryptography
and Data Security. Springer Berlin Heidelberg, 2013.*

[2] *publication in progress*

"""

import gmpy2
import time
import numpy as np

from typing import List, Tuple, Union, Callable
from ._jls_utils import invert, powmod
from gmpy2 import mpz, gcd
from math import ceil, floor, log2
from Crypto.Hash import SHA256
from ._jls_utils import TARGET_RANGE


class VEParameters:
    """Constants class for vector encoder parameters"""

    KEY_SIZE: int = 2048
    VALUE_SIZE: int = ceil(log2(10000))


class VES:
    """The vector encoding class

    This class encodes and decodes vector elements to create smaller size vectors using a packing technique


    Attributes:
        _ptsize: The bit length of the plaintext (the number of bits of an element in the output vector)
        _valuesize: The bit length of an element of the input vector
        _vectorsize: The number of element of the input vector
    """

    def __init__(
            self,
            ptsize: int,
            valuesize: int,
    ) -> None:
        """Vector encoder constructor

        Args:
            ptsize: The bit length of the plaintext (the number of bits of an element in the output vector)
            valuesize: The bit length of an element of the input vector

        """

        self._ptsize = ptsize
        self._valuesize = valuesize

    def _get_elements_size_and_compression_ratio(
            self,
            add_ops: int
    ) -> Tuple[int, int]:
        """Gets element size and compression ratio by given additional operation count.

        Args:
            add_ops:
        """
        element_size = self._valuesize + ceil(log2(add_ops))
        comp_ratio = floor(self._ptsize / element_size)

        return element_size, comp_ratio

    def encode(
            self,
            V: List[int],
            add_ops: int
    ) -> List[gmpy2.mpz]:
        """Encode a vector to a smaller size vector

        Args:
            V:
            add_ops:

        Returns:

        """
        element_size, comp_ratio = self._get_elements_size_and_compression_ratio(add_ops)

        bs = comp_ratio
        e = []
        E = []
        for v in V:
            e.append(v)
            bs -= 1
            if bs == 0:
                E.append(self._batch(e, element_size))
                e = []
                bs = comp_ratio
        if e:
            E.append(self._batch(e, element_size))
        return E

    def decode(
            self,
            E: List[int],
            add_ops: int
    ) -> List[int]:
        """Decode a vector back to original size vector

        Args:
            E:
            add_ops:
        Returns:
            Decoded vector
        """

        element_size, _ = self._get_elements_size_and_compression_ratio(add_ops)
        V = []

        for e in E:
            for v in self._debatch(e, element_size):
                V.append(v)
        return V

    @staticmethod
    def _batch(
            V,
            element_size: int
    ):
        i = 0
        a = 0
        for v in V:
            a |= v << element_size * i
            i += 1
        return mpz(a)

    @staticmethod
    def _debatch(
            b: int,
            element_size: int
    ) -> List[int]:
        """
        """
        i = 1
        V = []
        bit = 0b1
        mask = 0b1
        for _ in range(element_size - 1):
            mask <<= 1
            mask |= bit

        while b != 0:
            v = mask & b
            V.append(int(v))
            b >>= element_size
        return V


class EncryptedNumber(object):
    """An encrypted number by one of the user keys .

    Attributes:
        param: The public parameters
        ciphertext: The integer value of the ciphertext
    """

    def __init__(self, param, ciphertext):
        """

        Args:
            param: The public parameters.
            ciphertext: The integer value of the ciphertext

        """
        self.public_param = param
        self.ciphertext = ciphertext

    def __add__(
            self,
            other: Union['EncryptedNumber', mpz]
    ) -> 'EncryptedNumber':

        """Adds given value to self

        Args:

        Returns:

        """
        if isinstance(other, EncryptedNumber):
            return self._add_encrypted(other)
        if isinstance(other, mpz):
            e = EncryptedNumber(self.public_param, other)
            return self._add_encrypted(e)

    def __iadd__(self, other):
        return self.__add__(other)

    def __radd__(self, value: Union['EncryptedNumber', mpz]):
        """Allows summing parameters using built-in `sum` method

        Args:
            value: Value to add. It can be an instance of `mpz` or EncryptedNumber
        """
        if value == 0:
            return self
        else:
            return self.__add__(value)

    def __repr__(self):
        """Encrypted number representation """

        estr = self.ciphertext.digits()
        return "<EncryptedNumber {}...{}>".format(estr[:5], estr[-5:])

    def _add_encrypted(self, other: Union['EncryptedNumber', mpz]) -> 'EncryptedNumber':
        """Base add operation for single encrypted integer

        Args:
            other: Value to be added

        Returns:
            Sum of self and other
        """

        if self.public_param != other.public_param:
            raise ValueError(
                "Attempted to add numbers encrypted against " "different parameters!"
            )

        return EncryptedNumber(
            self.public_param, self.ciphertext * other.ciphertext % self.public_param.n_square
        )


class JoyeLibert:
    """The Joye-Libert scheme. It consists of three Probabilistic Polynomial Time algorithms:
    `Protect`, and `Agg`.

    Attributes:
        _vector_encoder: The vector encoding/decoding scheme

    """

    def __init__(self):
        """Constructs the class"""

        self._vector_encoder = VES(
            ptsize=VEParameters.KEY_SIZE // 2,
            valuesize=VEParameters.VALUE_SIZE
        )

    def protect(self,
                public_param,
                sk_u,
                tau,
                x_u_tau,
                n_users,
                ) -> List[EncryptedNumber]:
        """ Protect user input with the user's secret key:

        \\(y_{u,\\tau} \\gets \\textbf{JL.Protect}(public_param,sk_u,\\tau,x_{u,\\tau})\\)

        This algorithm encrypts private inputs
        \\(x_{u,\\tau} \\in \\mathbb{Z}_N\\) for time period \\(\\tau\\)
        using secret key \\(sk_u \\in \\mathbb{Z}_N^2\\) . It outputs cipher \\(y_{u,\\tau}\\) such that:

        $$y_{u,\\tau} = (1 + x_{u,\\tau} N) H(\\tau)^{sk_u} \\mod N^2$$

        Args:
            public_param: The public parameters \\(public_param\\)
            sk_u: The user's secret key \\(sk_u\\)
            tau: The time period \\(\\tau\\)
            x_u_tau: The user's input \\(x_{u,\\tau}\\)
            n_users: Number of nodes/users that participates secure aggregation

        Returns:
                The protected input of type `EncryptedNumber` or a list of `EncryptedNumber`
        """
        if not isinstance(sk_u, UserKey):
            raise TypeError(f"Expected key for encryption type is UserKey. but got {type(sk_u)}")

        if sk_u.public_param != public_param:
            raise ValueError("Bad public parameter. The public parameter of user key does not match the "
                             "one given for encryption")

        x_u_tau = self._vector_encoder.encode(
            V=x_u_tau,
            add_ops=n_users
        )
        return sk_u.encrypt(x_u_tau, tau)

    def aggregate(
            self,
            sk_0,
            tau,
            list_y_u_tau
    ) -> List[int]:
        """Aggregates users protected inputs with the server's secret key


        \\(X_{\\tau} \\gets \\textbf{JL.Agg}(public_param, sk_0,\\tau, \\{y_{u,\\tau}\\}_{u \\in \\{1,..,n\\}})\\)

        This algorithm aggregates the \\(n\\) ciphers received at time period \\(\\tau\\) to obtain
        \\(y_{\\tau} = \\prod_1^n{y_{u,\\tau}}\\) and decrypts the result.
        It obtains the sum of the private inputs ( \\( X_{\\tau} = \\sum_{1}^n{x_{u,\\tau}} \\) )
        as follows:

        $$V_{\\tau} = H(\\tau)^{sk_0} \\cdot y_{\\tau} \\qquad \\qquad X_{\\tau} = \\frac{V_{\\tau}-1}{N} \\mod N$$

        Args:
            sk_0: The server's secret key \\(sk_0\\)
            tau: The time period \\(\\tau\\)
            list_y_u_tau: A list of the users' protected inputs \\(\\{y_{u,\\tau}\\}_{u \\in \\{1,..,n\\}}\\)

        Returns:
            The sum of the users' inputs of type `int`
        """

        if not isinstance(sk_0, ServerKey):
            raise ValueError("Key must be an instance of `ServerKey`")

        if not isinstance(list_y_u_tau, list) or not list_y_u_tau:
            raise ValueError("list_y_u_tau should be a non-empty list.")

        if not isinstance(list_y_u_tau[0], list):
            raise ValueError("list_y_u_tau should be a list that contains list of encrypted numbers")

        n_user = len(list_y_u_tau)

        sum_of_vectors: List[EncryptedNumber] = [sum(ep) for ep in zip(*list_y_u_tau)]

        decrypted_vector = sk_0.decrypt(sum_of_vectors, tau)

        return self._vector_encoder.decode(decrypted_vector, add_ops=n_user)


class PublicParam:
    """The public parameters for Joye-Libert Scheme.

    ## **Attributes**:
    -------------
    **n** : `gmpy2.mpz` --
        The modulus \\(N\\)

        _n_square** : `gmpy2.mpz` --
        The square of the modulus \\(N^2\\)

    **bits** : `int` --
        The number of bits of the modulus \\(N\\)

    **H** : `function` --
        The hash algorithm \\(H : \\mathbb{Z} \\rightarrow \\mathbb{Z}_{N^2}^{*}\\)
    """

    def __init__(
            self,
            n_modulus: gmpy2.mpz,
            bits: int,
            hashing_function: Callable
    ) -> None:
        """

        Args:
            n_modulus: The modulus \\(N\\)
            bits: The number of bits of the modulus \\(N\\)
            hashing_function: The hash algorithm \\(H : \\mathbb{Z} \\rightarrow \\mathbb{Z}_{N^2}^{*}\\)
        """

        self._n_modulus = n_modulus
        self._n_square = n_modulus * n_modulus
        self._bits = bits
        self._hashing_function = hashing_function

    @property
    def bits(self) -> int:
        """Gets bits size

        Returns:
            Bits size
        """
        return self._bits

    @property
    def n_modulus(self) -> gmpy2.mpz:
        """Gets N modulus.

        Returns:
            N modulus
        """
        return self._n_modulus

    @property
    def n_square(self) -> gmpy2.mpz:
        """Gets squared N modulus

        Returns:
            Squared N modulus
        """
        return self._n_square

    def hashing_function(self, val: int):
        """Applies hashing

        Args:
            val: Value for hashing
        """
        return self._hashing_function(val)

    def __eq__(
            self,
            other: 'PublicParam'
    ) -> bool:
        """Compares equality of two public parameter

        Returns:
            True if other Public param's n_modules is equal to `self._n_modulus`
        """
        return self._n_modulus == other.n_modulus

    def __repr__(self) -> str:
        """Representation of Public parameters

        Returns:
            Representation as string
        """
        hashcode = hex(hash(self._hashing_function))
        n_str = self._n_modulus.digits()
        return "<PublicParam (N={}...{}, H(x)={})>".format(n_str[:5], n_str[-5:], hashcode[:10])


class BaseKey:
    """A base key class for Joye-Libert Scheme.

    Attributes:
        _public_param: The public parameter
        _key: Server key

    """

    def __init__(self, param: PublicParam, key: int):
        """Constructs base key object
        Args:
            param: The public parameters
            key: The value of the server's key \\(sk_0\\)
        """

        if not isinstance(key, int):
            raise TypeError("The key should be type of integer")

        self._public_param = param
        self._key = mpz(key)

    @property
    def public_param(
            self
    ) -> PublicParam:
        """Return public parameter of the key"""
        return self._public_param

    def __repr__(self):
        """Representation of ServerKey object"""
        hashcode = hex(hash(self))
        return "<ServerKey {}>".format(hashcode[:10])

    def __eq__(
            self,
            other: 'ServerKey'
    ) -> bool:
        """Check equality of public parameters

        Args:
            other: other server key object to compare.

        Returns:
            True if both ServerKey uses same public params. False for vice versa.
        """
        return self._public_param == other.public_param and self._key == other.s

    def __hash__(self) -> str:
        """Hash of server key"""
        return hash(self._key)

    def _populate_tau(self, tau: int, len_: int, ):
        """Populates TAU by applying hashing function"""

        taus = (np.arange(0, len_) << self._public_param.bits // 2) | tau

        # with multiprocessing.Pool() as pool:
        #     result = pool.map(self._public_param.hashing_function, taus)

        return [self._public_param.hashing_function(t) for t in taus]


class UserKey(BaseKey):
    """A user key for Joye-Libert Scheme. """

    def __init__(
            self,
            public_param: PublicParam,
            key: int
    ) -> None:
        """Server key constructor.

        Args:
            public_param: The public parameters
            key: The value of the server's key \\(sk_0\\)
        """
        super().__init__(public_param, key)

    def encrypt(
            self,
            plaintext: List[gmpy2.mpz],
            tau: int
    ):
        """Encrypts a plaintext  for time period tau

        Args:
            plaintext: The plaintext/value to encrypt
            tau:  The time period
           

        Returns:
            A ciphertext of the plaintext, encrypted by the user key of type `EncryptedNumber` or list of
                encrypted numbers.

        """
        if not isinstance(plaintext, list):
            raise TypeError(f"Expected plaintext type list but got {type(plaintext)}")

        # TODO: find-out what is going wrong in numpy implementation
        # Use numpy vectors to increase speed of calculation
        # plaintext = np.array(plaintext)
        # nude_ciphertext = (self._public_param.n_modulus * (plaintext + 1)) % self._public_param.n_modulus
        # taus = self._populate_tau(tau=tau, len_=len(plaintext))
        #
        # # This process takes some time
        # vec_pow_mod = np.vectorize(powmod, otypes=[mpz])
        # r = vec_pow_mod(taus, self._key, self._public_param.n_square)
        # cipher = (nude_ciphertext * r) % self._public_param.n_square
        # cipher = [EncryptedNumber(self._public_param, ciphertext) for ciphertext in cipher]

        # TODO: Remove old implementation
        cipher = [self._encrypt(pt, (i << self._public_param.bits // 2) | tau)
                  for i, pt in plaintext]

        return cipher

    def _encrypt(
            self,
            plaintext: gmpy2.mpz,
            tau: int
    ) -> EncryptedNumber:
        """Encrypts given single plaintext as int

        Args:
            plaintext: The plaintext/value to encrypt
            tau:  The time period

        Returns:
            A ciphertext of the plaintext, encrypted by the user key of type `EncryptedNumber`

        """
        nude_ciphertext = (self._public_param.n_modulus * plaintext + 1) % self._public_param.n_square
        r = powmod(self._public_param.hashing_function(tau), self._key, self._public_param.n_square)
        ciphertext = (nude_ciphertext * r) % self._public_param.n_square

        return EncryptedNumber(self._public_param, ciphertext)


class ServerKey(BaseKey):
    """A server key for Joye-Libert Scheme. """

    def __init__(
            self,
            public_param: PublicParam,
            key: int
    ) -> None:
        """Server key constructor.

        Args:
            public_param: The public parameters
            key: The value of the server's key \\(sk_0\\)
        """
        super().__init__(public_param, key)

    def decrypt(
            self,
            cipher: List[EncryptedNumber],
            tau: int,
            delta: int = 1
    ) -> List[int]:
        """Decrypts the aggregated ciphertexts of all users for time period tau

        Args:
            cipher:  An aggregated ciphertext, with weighted averaging or not
            tau: The time period, (training round)
            delta: ...

        Raises:
            TypeError: In case of invalid argument type.

        Returns:
            List of decrypted sum of user inputs
        """

        if not isinstance(cipher, list):
            raise TypeError(f"Expected `cipher` is list of encrypter numbers but got {type(cipher)}")

        if not all([isinstance(c, EncryptedNumber) for c in cipher]):
            raise TypeError(f"Cipher text should be list of EncryptedNumbers")

        # TODO: Findout what is going wrong in numpy implementation
        # ciphertext = [c.ciphertext for c in cipher]
        # taus = self._populate_tau(tau=tau, len_=len(ciphertext))
        #
        # powmod_ = np.vectorize(powmod, otypes=[mpz])
        # mod = powmod_(taus,  delta ** 2 * self._key, self._public_param.n_square)
        #
        # v = (ciphertext * mod) % self._public_param.n_square
        # x = ((v - 1) // self._public_param.n_modulus) % self._public_param.n_modulus
        #
        # inverted = invert(delta ** 2, self._public_param.n_square)
        #
        # x = (x * inverted) % self._public_param.n_modulus
        #
        # pt = [int(pt) for pt in x]

        # TODO: Remove old implementation
        pt = [self._decrypt(c, (i << self._public_param.bits // 2) | tau, delta)
                  for i, c in enumerate(cipher)]

        return pt

    # TODO: Remove this method before merging
    def _decrypt(
            self,
            cipher: EncryptedNumber,
            tau: int,
            delta=1
    ) -> int:
        """Decrypts given single encrypted number using server key

        Args:
            cipher:  An aggregated ciphertext, with weighted averaging or not
            tau: The time period, (training round)
            delta: ...

        Returns:
            Decrypted sum of user inputs of type `int`
        """
        if not isinstance(cipher, EncryptedNumber):
            raise TypeError("Expected encrypted number type but got: %s" % type(cipher))

        if self._public_param != cipher.public_param:
            raise ValueError(
                "encrypted_number was encrypted against a " "different key!"
            )

        cipher_text = cipher.ciphertext

        if not isinstance(cipher_text, mpz):
            raise TypeError(
                "Expected mpz type ciphertext but got: %s" % type(cipher_text)
            )

        v = (cipher_text *
             powmod(self._public_param.hashing_function(tau),
                    delta ** 2 * self._key,
                    self._public_param.n_square
                    )
             ) % self._public_param.n_square
        x = ((v - 1) // self._public_param.n_modulus) % self._public_param.n_modulus
        x = (x * invert(delta ** 2, self._public_param.n_square)) % self._public_param.n_modulus

        return int(x)


class FDH:
    """The Full-Domain Hash scheme.

    This class computes a full domain hash value using SHA256 hash function
    """

    def __init__(
            self,
            bits_size: int,
            n_modulus: gmpy2.mpz
    ) -> None:
        """Constructs FDH.

        Args:
            bits_size: The bit length of the output of the FDH
            n_modulus: The modulus \\(N\\) such that the FDH output is in \\(\\mathbb{Z}^*_N\\)
        """

        if not isinstance(bits_size, int):
            raise TypeError(f"Bits size should be an integer not {type(bits_size)}")

        if not isinstance(n_modulus, gmpy2.mpz):
            raise TypeError(f"n_modules should be of type `gmpy2.mpz` not {type(n_modulus)}")

        self.bits_size = bits_size
        self._n_modules = n_modulus

    def H(
            self,
            t: int
    ) -> gmpy2.mpz:

        """Computes the FDH using SHA256.

        !!! infor "Computation"
            $$\\textbf{SHA256}(x||0) ||\\textbf{SHA256}(x||1) || ... || \\textbf{SHA256}(x||c) \\mod N$$
                where \\(c\\) is a counter that keeps incrementing until the size of the output has *bits_size*
                length and the output falls in  \\(\\mathbb{Z}^*_N\\)

        Args:
            t: The input of the hash function

        Returns:
            A value in \\(\\mathbb{Z}^*_N\\)
        """

        counter = 1
        result = b""

        while True:
            while True:
                h = SHA256.new()
                h.update(
                    int(t).to_bytes(self.bits_size // 2, "big")
                    + counter.to_bytes(1, "big")
                )
                result += h.digest()
                counter += 1
                if len(result) < (self.bits_size // 8):
                    break

            r = mpz(int.from_bytes(result[-self.bits_size:], "big"))

            if gcd(r, self._n_modules) == 1:
                break

        return r
