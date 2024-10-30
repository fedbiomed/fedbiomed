"""
Joye-Libert secure aggregation scheme (JL) and its threshold-variant (TJL)

This module contains a python implementation of the Joye-Libert scheme and the
threshold-variant of Joye-Libert scheme. The original scheme of Joye-Libert can be
found here [1]. The threshold variant is defined here [2].

**Implemented by:** Mohamad Mansouri (mohamad.mansouri@thalesgroup.com) under MIT License

[1] *Marc Joye and Benoît Libert. A scalable scheme for privacy-preserving aggregation
of time-series data. In Ahmad-Reza Sadeghi, editor, Financial Cryptography
and Data Security. Springer Berlin Heidelberg, 2013.*

[2] *publication in progress* (to be updated)


**Note:** In this module, Mohamad Mansouri's original implementation has been slightly modified to
comply with coding conventions (annotations, typing, doc strings, argument/variable names, etc.).
Also, some changes have been made to make the original implementation compatible with the
Fed-BioMed FL infrastructure.
"""

import hashlib
from math import ceil, floor, log2
from typing import List, Tuple, Union, Callable

import gmpy2
from gmpy2 import mpz, gcd
import numpy as np

from fedbiomed.common.constants import SAParameters


def invert(
        a: mpz,
        b: mpz
) -> mpz:
    """Finds the inverts of a mod b

    Args:
        a: number to invert
        b: modulo of inversion

    Returns:
        inverted value

    Raises:
        ZeroDivisionError: cannot invert number
    """

    s = gmpy2.invert(a, b)
    # according to documentation, gmpy2.invert might return 0 on
    # non-invertible element, although it seems to actually raise an
    # exception; for consistency, we always raise the exception
    if s == 0:
        raise ZeroDivisionError("invert() no inverse exists")
    return s


def powmod(
        a: mpz,
        b: mpz,
        c: mpz
) -> mpz:
    """Computes a to the power of b mod c

    Args:
        a: number to compute power
        b: power
        c: power modulus

    Returns:
        powered value of a
    """
    if a == 1:
        return 1
    return gmpy2.powmod(a, b, c)




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

        self._ptsize: int = ptsize
        self._valuesize: int = valuesize

    def _get_elements_size_and_compression_ratio(
            self,
            add_ops: int
    ) -> Tuple[int, int]:
        """Gets element size and compression ratio by given additional operation count.

        Args:
            add_ops: ?

        Returns:
            tuple of element size and compression ratio
        """
        element_size = self._valuesize + ceil(log2(add_ops + 1))
        comp_ratio = floor(self._ptsize / element_size)

        return element_size, comp_ratio

    def encode(
            self,
            V: List[int],
            add_ops: int
    ) -> List[gmpy2.mpz]:
        """Encode a vector to a smaller size vector

        Args:
            V: ?
            add_ops: ?

        Returns:
            list of encoded values
        """
        element_size, comp_ratio = self._get_elements_size_and_compression_ratio(add_ops)

        bs = comp_ratio
        e = []
        E = []
        for v in V:
            e.append(v)
            bs -= 1
            if bs == 0:  # rather use `if bs < 1`
                E.append(self._batch(e, element_size))
                e = []
                bs = comp_ratio
        if e:
            E.append(self._batch(e, element_size))
        return E

    def decode(
            self,
            E: List[int],
            add_ops: int,
            v_expected: int
    ) -> List[int]:
        """Decode a vector back to original size vector

        Args:
            E: encoded parameters to decode
            add_ops: ?
            v_expected: number of parameters to decode from the encoded parameters
        Returns:
            Decoded vector
        """

        element_size, comp_ratio = self._get_elements_size_and_compression_ratio(add_ops)
        V = []

        for e in E:
            v_number = min(v_expected, comp_ratio)
            for v in self._debatch(e, element_size, v_number):
                V.append(v)
            v_expected -= v_number
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
            element_size: int,
            element_number: int
    ) -> List[int]:
        """
        """
        V = []
        bit = 0b1
        mask = 0b1
        for _ in range(element_size - 1):
            mask <<= 1
            mask |= bit

        for _ in range(element_number):
            v = mask & b
            V.append(int(v))
            b >>= element_size
        return V


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
            n_modulus: mpz,
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

        Returns:
            hashed value
        """
        return self._hashing_function(val)

    def __eq__(
            self,
            other: 'PublicParam'
    ) -> bool:
        """Compares equality of two public parameter

        Args:
            other: other PublicParam

        Returns:
            True if other PublicParam's n_modules is equal to `self._n_modulus`
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


class EncryptedNumber(object):
    """An encrypted number by one of the user keys .

    Attributes:
        param: The public parameters
        ciphertext: The integer value of the ciphertext
    """

    def __init__(self, param: PublicParam, ciphertext: int):
        """

        Args:
            param: The public parameters.
            ciphertext: The integer value of the ciphertext

        """
        self.public_param = param
        self.ciphertext = mpz(ciphertext)

    def __add__(
            self,
            other: 'EncryptedNumber'
    ) -> 'EncryptedNumber':

        """Adds given value to self

        Args:
            other: value to add to self

        Returns:
            summed value
        """

        if not isinstance(other, EncryptedNumber):
            raise TypeError("Encrypted number can be only summed with another Encrypted num."
                            f"Can not sum Encrypted number with type {type(other)}")

        return self._add_encrypted(other)

    def __iadd__(self, other):
        return self.__add__(other)

    def __radd__(self, other: Union['EncryptedNumber', int]) -> 'EncryptedNumber':
        """Allows summing parameters using built-in `sum` method

        Args:
            other: Value to add. It should be an instance EncryptedNumber

        Returns:
            summed value
        """
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __repr__(self) -> str:
        """Encrypted number representation

        Returns:
            encypted number
        """

        repr = self.ciphertext.digits()
        return "<EncryptedNumber {}...{}>".format(repr[:5], repr[-5:])

    def _add_encrypted(
            self,
            other: 'EncryptedNumber'
    ) -> 'EncryptedNumber':
        """Base add operation for single encrypted integer

        Args:
            other: Value to be added

        Returns:
            Sum of self and other

        Raises:
            ValueError: summing numbers encrypted against different parameters
        """

        if self.public_param != other.public_param:
            raise ValueError(
                "Attempted to add numbers encrypted against " "different parameters!"
            )

        return EncryptedNumber(
            self.public_param,
            self.ciphertext * other.ciphertext % self.public_param.n_square
        )


class BaseKey:
    """A base key class for Joye-Libert Scheme.

    Attributes:
        _public_param: The public parameter
        _key: Server key

    """

    def __init__(self, public_param: PublicParam, key: int):
        """Constructs base key object

        Args:
            param: The public parameters
            key: The value of the server's key \\(sk_0\\)

        Raises:
            TypeError: bad argument type
        """

        if not isinstance(key, int):
            raise TypeError("The key should be type of integer")

        self._public_param = public_param
        self._key = mpz(key)

    @property
    def public_param(
            self
    ) -> PublicParam:
        """Return public parameter of the key

        Returns:
            public param
        """
        return self._public_param

    @property
    def key(
            self
    ) -> mpz:
        """Gets the key.

        Returns:
            The user or server key
        """
        return self._key

    def __repr__(self):
        """Representation of ServerKey object"""
        hashcode = hex(hash(self))
        return "<ServerKey {}>".format(hashcode[:10])

    def __eq__(
            self,
            other: Union['BaseKey', 'ServerKey', 'UserKey']
    ) -> bool:
        """Check equality of public parameters

        Args:
            other: other server key object to compare.

        Returns:
            True if both ServerKey uses same public params. False for vice versa.

        Raises:
            TypeError: bad argument type
        """
        if not isinstance(other, type(self)):
            raise TypeError(f"The key can not be compared with type {type(other)}")

        return self._public_param == other.public_param and self._key == other.key

    def __hash__(self) -> str:
        """Hash of server key

        Returns:
            hash value
        """
        return hash(self._key)

    def _populate_tau(self, tau: int, len_: int, ):
        """Populates TAU by applying hashing function

        Args:
            tau: `tau` value to use for generating list numbers
            len: number of elements in tau list

        Returns:
            list of tau values
        """
        taus = (np.arange(0, len_, dtype=mpz) << self._public_param.bits // 2) | tau

        return [self._public_param.hashing_function(t) for t in taus]


class UserKey(BaseKey):
    """A user key for Joye-Libert Scheme. """

    def encrypt(
            self,
            plaintext: List[mpz],
            tau: int
    ) -> List[mpz]:
        """Encrypts a plaintext  for time period tau

        Args:
            plaintext: The plaintext/value to encrypt
            tau:  The time period


        Returns:
            A ciphertext of the plaintext, encrypted by the user key of
                type `EncryptedNumber` or list of encrypted numbers.

        Raises:
            TypeError: bad parameters type
        """
        if not isinstance(plaintext, list):
            raise TypeError(f"Expected plaintext type list but got {type(plaintext)}")

        # TODO: find-out what is going wrong in numpy implementation
        # Use numpy vectors to increase speed of calculation
        plaintext = np.array(plaintext)
        nude_ciphertext = \
            (self._public_param.n_modulus * plaintext + 1) \
            % self._public_param.n_square
        taus = self._populate_tau(tau=tau, len_=len(plaintext))

        # This process takes some time
        vec_pow_mod = np.vectorize(powmod, otypes=[mpz])
        r = vec_pow_mod(taus, self._key, self._public_param.n_square)
        cipher = (nude_ciphertext * r) % self._public_param.n_square

        # Convert np array to list
        return cipher.tolist()


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

        Returns:
            List of decrypted sum of user inputs

        Raises:
            TypeError: In case of invalid argument type.

        """

        if not isinstance(cipher, list):
            raise TypeError(f"Expected `cipher` is list of encrypter numbers but got {type(cipher)}")

        if not all([isinstance(c, EncryptedNumber) for c in cipher]):
            raise TypeError("Cipher text should be list of EncryptedNumbers")

        # TODO: Find out what is going wrong in numpy implementation
        ciphertext = [c.ciphertext for c in cipher]
        taus = self._populate_tau(tau=tau, len_=len(ciphertext))

        powmod_ = np.vectorize(powmod, otypes=[mpz])
        mod = powmod_(taus, delta ** 2 * self._key, self._public_param.n_square)

        v = (ciphertext * mod) % self._public_param.n_square
        x = ((v - 1) // self._public_param.n_modulus) % self._public_param.n_modulus

        inverted = invert(delta ** 2, self._public_param.n_square)

        x = (x * inverted) % self._public_param.n_modulus

        pt = [int(pt) for pt in x]

        return pt


class JoyeLibert:
    """The Joye-Libert scheme. It consists of three Probabilistic Polynomial Time algorithms:
    `Protect`, and `Agg`.

    Attributes:
        _vector_encoder: The vector encoding/decoding scheme

    """

    def __init__(self):
        """Constructs the class

        VEParameters.TARGET_RANGE + VEParameters.WEIGHT_RANGE should be
        equal or less than 2**32
        """
        self._vector_encoder = VES(
            ptsize=SAParameters.KEY_SIZE // 2,
            valuesize=ceil(log2(SAParameters.TARGET_RANGE) + log2(SAParameters.WEIGHT_RANGE))
        )

    def protect(self,
                public_param: PublicParam,
                user_key: UserKey,
                tau: int,
                x_u_tau: List[int],
                n_users: int,
                ) -> List[mpz]:
        """ Protect user input with the user's secret key:

        \\(y_{u,\\tau} \\gets \\textbf{JL.Protect}(public_param,sk_u,\\tau,x_{u,\\tau})\\)

        This algorithm encrypts private inputs
        \\(x_{u,\\tau} \\in \\mathbb{Z}_N\\) for time period \\(\\tau\\)
        using secret key \\(sk_u \\in \\mathbb{Z}_N^2\\) . It outputs cipher \\(y_{u,\\tau}\\) such that:

        $$y_{u,\\tau} = (1 + x_{u,\\tau} N) H(\\tau)^{sk_u} \\mod N^2$$

        Args:
            public_param: The public parameters \\(public_param\\)
            user_key: The user's secret key \\(sk_u\\)
            tau: The time period \\(\\tau\\)
            x_u_tau: The user's input \\(x_{u,\\tau}\\)
            n_users: Number of nodes/users that participates secure aggregation

        Returns:
                The protected input of type `EncryptedNumber` or a list of `EncryptedNumber`

        Raises:
                TypeError: bad argument type
                ValueError: bad argument value
        """
        if not isinstance(user_key, UserKey):
            raise TypeError(f"Expected key for encryption type is UserKey. but got {type(user_key)}")

        if user_key.public_param != public_param:
            raise ValueError("Bad public parameter. The public parameter of user key does not match the "
                             "one given for encryption")

        if not isinstance(x_u_tau, list):
            raise TypeError(f"Bad vector for encryption. Excepted argument `x_u_tau` type list but "
                            f"got {type(x_u_tau)}")

        x_u_tau = self._vector_encoder.encode(
            V=x_u_tau,
            add_ops=n_users
        )

        return user_key.encrypt(x_u_tau, tau)

    def aggregate(
            self,
            sk_0: ServerKey,
            tau: int,
            list_y_u_tau: List[List[EncryptedNumber]],
            num_expected_params: int
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
            num_expected_params: Number of parameters to decode from the decrypted vectors

        Returns:
            The sum of the users' inputs of type `int`

        Raises:
            ValueError: bad argument value
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

        return self._vector_encoder.decode(decrypted_vector, add_ops=n_user, v_expected=num_expected_params)


class FDH:
    """The Full-Domain Hash scheme.

    This class computes a full domain hash value using SHA256 hash function
    """

    def __init__(
            self,
            bits_size: int,
            n_modulus: mpz
    ) -> None:
        """Constructs FDH.

        Args:
            bits_size: The bit length of the output of the FDH
            n_modulus: The modulus \\(N\\) such that the FDH output is in \\(\\mathbb{Z}^*_N\\)
        """

        if not isinstance(bits_size, int):
            raise TypeError(f"Bits size should be an integer not {type(bits_size)}")

        if not isinstance(n_modulus, mpz):
            raise TypeError(f"n_modules should be of type `gmpy2.mpz` not {type(n_modulus)}")

        self.bits_size = bits_size
        self._n_modules = n_modulus

    def H(
            self,
            t: int
    ) -> mpz:
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
                h = hashlib.sha256()
                h.update(
                    int(t).to_bytes(self.bits_size // 2, "big") +
                    counter.to_bytes(1, "big")
                )
                result += h.digest()
                counter += 1
                if len(result) < (self.bits_size // 8):
                    break

            r = mpz(int.from_bytes(result[-self.bits_size:], "big"))

            if gcd(r, self._n_modules) == 1:
                break

        return r
