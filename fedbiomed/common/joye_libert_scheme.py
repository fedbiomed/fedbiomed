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

import random

from gmpy2 import mpz

from fedbiomed.common.joye_libert_utils import getprimeover, invert, powmod

DEFAULT_KEY_SIZE = 2048


class JLS(object):
    """
    The Joye-Libert scheme. It consists of three Probabilistic Polynomial Time algorithms: **Setup**,
    **Protect**, and **Agg**.

    ## **Args**:
    -------------
    *nusers* : `int` --
        The number of users in the scheme

    *VE* : `VectorEncoding` --
        The vector encoding/decoding scheme (default: `None`)

    ## **Attributes**:
    -------------
    *nusers* : `int` --
        The number of users in the scheme

    *VE* : `VectorEncoding` --
        The vector encoding/decoding scheme

    *keysize* : `int` --
        The bit length of the keys



    """

    def __init__(self, nusers, VE=None):
        super().__init__()
        self.nusers = nusers
        self.keysize = None
        self.VE = VE

    def Setup(self, lmbda=DEFAULT_KEY_SIZE):
        """
        Setups the users and the server with the secret keys and public parameters

        ### Given some security parameter \\(\\lambda\\), this algorithm generates two equal-size prime numbers \\(p\\) and \\(q\\) and sets \\(N = pq\\). It randomly generates \\(n\\) secret keys \\(sk_u \\xleftarrow{R} \\pm \\{0,1\\}^{2l}\\) where \\(l\\) is the number of bits of \\(N\\) and sets \\(sk_0 = -\\sum_{1}^n{sk_u}\\). Then, it defines a cryptographic hash function \\(H : \\mathbb{Z} \\rightarrow \\mathbb{Z}_{N^2}^{*}\\). It outputs the \\(n+1\\) keys and the public parameters \\((N, H)\\).

        ## **Args**:
        -------------
        **lmbda** : `int` --
            The bit length the user/server key

        ## **Returns**:
        -------------
        The public parameters, server key, a list of user keys:  `(PublicParam, ServerKey, list[UserKeys])`
        """
        self.keysize = lmbda

        p = q = n = None
        n_len = 0
        # while n_len != lmbda // 2:
        #     p = mpz(7801876574383880214548650574033350741129913580793719706746361606042541080141291132224899113047934760791108387050756752894517232516965892712015132079112571) #getprimeover(lmbda // 4)
        #     q = p
        #     while q == p:
        #         q = mpz(7755946847853454424709929267431997195175500554762787715247111385596652741022399320865688002114973453057088521173384791077635017567166681500095602864712097) #getprimeover(lmbda // 4)
        p = mpz(
            7801876574383880214548650574033350741129913580793719706746361606042541080141291132224899113047934760791108387050756752894517232516965892712015132079112571
        )  # getprimeover(lmbda // 4)
        q = mpz(
            7755946847853454424709929267431997195175500554762787715247111385596652741022399320865688002114973453057088521173384791077635017567166681500095602864712097
        )  # getprimeover(lmbda // 4)

        n = p * q
        n_len = n.bit_length()
        print("n_len", n_len)
        fdh = FDH(self.keysize, n * n)

        public_param = PublicParam(n, lmbda // 2, fdh.H)

        seed = random.SystemRandom()
        s0 = mpz(0)
        # users = {}
        #
        # for i in range(self.nusers):
        #     s = mpz(0)  # s = mpz(seed.getrandbits(2 * n_len))
        #     users[i] = UserKey(public_param, s)
        #     s0 += s
        # s0 = -s0
        server = ServerKey(public_param, s0)
        user = UserKey(public_param, s0)
        return public_param, server, user

    def Protect(self, pp, sk_u, tau, x_u_tau):
        """
        Protect user input with the user's secret key: \\(y_{u,\\tau} \\gets \\textbf{JL.Protect}(pp,sk_u,\\tau,x_{u,\\tau})\\)

        ### This algorithm encrypts private inputs \\(x_{u,\\tau} \\in \\mathbb{Z}_N\\) for time period \\(\\tau\\) using secret key \\(sk_u \\in \\mathbb{Z}_N^2\\) . It outputs cipher \\(y_{u,\\tau}\\) such that:

        $$y_{u,\\tau} = (1 + x_{u,\\tau} N) H(\\tau)^{sk_u} \\mod N^2$$

        ## **Args**:
        -------------

        *pp* : `PublicParam` --
            The public parameters \\(pp\\)

        *sk_u* : `UserKey` --
            The user's secret key \\(sk_u\\)

        *tau* : `int` --
            The time period \\(\\tau\\)

        *x_u_tau* : `int` or `list` --
            The user's input \\(x_{u,\\tau}\\)

        ## **Returns**:
        -------------
        The protected input of type `EncryptedNumber` or a list of `EncryptedNumber`
        """
        assert isinstance(sk_u, UserKey), "bad user key"
        assert sk_u.pp == pp, "bad user key"

        if isinstance(x_u_tau, list):
            x_u_tau = self.VE.encode(x_u_tau)
            return sk_u.encrypt(x_u_tau, tau)
        else:
            return sk_u.encrypt(x_u_tau, tau)

    def Agg(self, pp, sk_0, tau, list_y_u_tau):
        """
        Aggregate users protected inputs with the server's secret key: \\(X_{\\tau} \\gets \\textbf{JL.Agg}(pp, sk_0,\\tau, \\{y_{u,\\tau}\\}_{u \\in \\{1,..,n\\}})\\)

        ### This algorithm aggregates the \\(n\\) ciphers received at time period \\(\\tau\\) to obtain \\(y_{\\tau} = \\prod_1^n{y_{u,\\tau}}\\) and decrypts the result. It obtains the sum of the private inputs ( \\( X_{\\tau} = \\sum_{1}^n{x_{u,\\tau}} \\) ) as follows:

        $$V_{\\tau} = H(\\tau)^{sk_0} \\cdot y_{\\tau} \\qquad \\qquad X_{\\tau} = \\frac{V_{\\tau}-1}{N} \\mod N$$

        ## **Args**:
        -------------

        *pp* : `PublicParam` --
            The public parameters \\(pp\\)

        *sk_0* : `ServerKey` --
            The server's secret key \\(sk_0\\)

        *tau* : `int` --
            The time period \\(\\tau\\)

        *list_y_u_tau* : `list` --
            A list of the users' protected inputs \\(\\{y_{u,\\tau}\\}_{u \\in \\{1,..,n\\}}\\)

        ## **Returns**:
        -------------
        The sum of the users' inputs of type `int`
        """
        assert isinstance(sk_0, ServerKey), "bad server key"
        # assert sk_0.pp == pp, "bad server key"
        assert isinstance(list_y_u_tau, list), "list_y_u_tau should be a list"
        assert (
            len(list_y_u_tau) > 0
        ), "list_y_u_tau should contain at least one protected input"
        if isinstance(list_y_u_tau[0], list):
            for y_u_tau in list_y_u_tau:
                assert len(list_y_u_tau[0]) == len(
                    y_u_tau
                ), "attempting to aggregate protected vectors of different sizes"
            y_tau = []
            for i in range(len(list_y_u_tau[0])):
                y_tau_i = list_y_u_tau[0][i]
                for y_u_tau in list_y_u_tau[1:]:
                    y_tau_i += y_u_tau[i]
                y_tau.append(y_tau_i)
            d = sk_0.decrypt(y_tau, tau)
            sum_x_u_tau = self.VE.decode(d)

        else:
            assert isinstance(list_y_u_tau[0], EncryptedNumber), "bad ciphertext"
            y_tau = list_y_u_tau[0]
            for y_u_tau in list_y_u_tau[1:]:
                y_tau += y_u_tau
            sum_x_u_tau = sk_0.decrypt(y_tau, tau)

        return sum_x_u_tau


class PublicParam(object):
    """
    The public parameters for Joye-Libert Scheme.

    ## **Args**:
    -------------
    **n** : `gmpy2.mpz` --
        The modulus \\(N\\)

    **bits** : `int` --
        The number of bits of the modulus \\(N\\)

    **H** : `function` --
        The hash algorithm \\(H : \\mathbb{Z} \\rightarrow \\mathbb{Z}_{N^2}^{*}\\)


    ## **Attributes**:
    -------------
    **n** : `gmpy2.mpz` --
        The modulus \\(N\\)

    **nsquare** : `gmpy2.mpz` --
        The square of the modulus \\(N^2\\)

    **bits** : `int` --
        The number of bits of the modulus \\(N\\)

    **H** : `function` --
        The hash algorithm \\(H : \\mathbb{Z} \\rightarrow \\mathbb{Z}_{N^2}^{*}\\)
    """

    def __init__(self, n, bits, H):
        super().__init__()
        self.n = n
        self.nsquare = n * n
        self.bits = bits
        self.H = H

    def __eq__(self, other):
        return self.n == other.n

    def __repr__(self):
        hashcode = hex(hash(self.H))
        nstr = self.n.digits()
        return "<PublicParam (N={}...{}, H(x)={})>".format(
            nstr[:5], nstr[-5:], hashcode[:10]
        )


class UserKey(object):
    """
    A user key for Joye-Libert Scheme.

    ## **Args**:
    -------------
    **param** : `PublicParam` --
        The public parameters

    **key** : `gmpy2.mpz` --
        The value of the user's key \\(sk_0\\)

    ## **Attributes**:
    -------------
    **param** : `PublicParam` --
        The public parameters

    **key** : `gmpy2.mpz` --
        The value of the user's key \\(sk_0\\)
    """

    def __init__(self, param, key):
        super().__init__()
        self.pp = param
        self.s = key

    def __repr__(self):
        hashcode = hex(hash(self))
        return "<UserKey {}>".format(hashcode[:10])

    def __eq__(self, other):
        return self.pp == other.pp and self.s == other.s

    def __hash__(self):
        return hash(self.s)

    def encrypt(self, plaintext, tau):
        """
        Encrypts a plaintext  for time period tau

        ## **Args**:
        -------------
        **plaintext** : `int` or `gmpy2.mpz` --
            the plaintext to encrypt

        **tau** : `int` --
            the time period

        ## **Returns**:
        ---------------
        A ciphertext of the *plaintext* encrypted by the user key of type `EncryptedNumber`
        """
        if isinstance(plaintext, list):
            counter = 0
            cipher = []
            for pt in plaintext:
                cipher.append(self._encrypt(pt, (counter << self.pp.bits // 2) | tau))
                counter += 1
        else:
            cipher = self._encrypt(plaintext, tau)
        return cipher

    def _encrypt(self, plaintext, tau):
        nude_ciphertext = (self.pp.n * plaintext + 1) % self.pp.nsquare
        r = powmod(self.pp.H(tau), self.s, self.pp.nsquare)
        ciphertext = (nude_ciphertext * r) % self.pp.nsquare
        return EncryptedNumber(self.pp, ciphertext)


class ServerKey(object):
    """
    A server key for Joye-Libert Scheme.

    ## **Args**:
    -------------
    **param** : `PublicParam` --
        The public parameters

    **key** : `gmpy2.mpz` --
        The value of the server's key \\(sk_0\\)

    ## **Attributes**:
    -------------
    **param** : `PublicParam` --
        The public parameters

    **key** : `gmpy2.mpz` --
        The value of the server's key \\(sk_0\\)
    """

    def __init__(self, param, key):
        super().__init__()
        self.pp = param
        self.s = key

    def __repr__(self):
        hashcode = hex(hash(self))
        return "<ServerKey {}>".format(hashcode[:10])

    def __eq__(self, other):
        return self.pp == other.pp and self.s == other.s

    def __hash__(self):
        return hash(self.s)

    def decrypt(self, cipher, tau, delta=1):
        """
        Decrypts the aggregated ciphertexts of all users for time period tau

        ## **Args**:
        -------------
        **cipher** : `EncryptedNumber` --
            An aggregated ciphertext

        **tau** : `int` --
            the time period

        ## **Returns**:
        ---------------
        The sum of user inputs of type `int`
        """

        if isinstance(cipher, list):
            counter = 0
            pt = []
            for c in cipher:
                pt.append(self._decrypt(c, (counter << self.pp.bits // 2) | tau, delta))
                counter += 1
        else:
            pt = self._decrypt(cipher, tau, delta)
        return pt

    def _decrypt(self, cipher, tau, delta=1):
        if not isinstance(cipher, EncryptedNumber):
            raise TypeError("Expected encrypted number type but got: %s" % type(cipher))
        if self.pp != cipher.pp:
            raise ValueError(
                "encrypted_number was encrypted against a " "different key!"
            )
        return self._raw_decrypt(cipher.ciphertext, tau, delta)

    def _raw_decrypt(self, ciphertext, tau, delta=1):
        if not isinstance(ciphertext, mpz):
            raise TypeError(
                "Expected mpz type ciphertext but got: %s" % type(ciphertext)
            )
        V = (
            ciphertext * powmod(self.pp.H(tau), delta**2 * self.s, self.pp.nsquare)
        ) % self.pp.nsquare
        X = ((V - 1) // self.pp.n) % self.pp.n
        X = (X * invert(delta**2, self.pp.nsquare)) % self.pp.n
        return int(X)


class EncryptedNumber(object):
    """
    An encrypted number by one of the user keys .

    ## **Args**:
    -------------
    **param** : `PublicParam` --
        The public parameters

    **ciphertext** : `gmpy2.mpz` --
        The integer value of the ciphertext

    ## **Attributes**:
    -------------
    **param** : `PublicParam` --
        The public parameters

    **ciphertext** : `gmpy2.mpz` --
        The integer value of the ciphertext
    """

    def __init__(self, param, ciphertext):
        super().__init__()
        self.pp = param
        self.ciphertext = ciphertext

    def __add__(self, other):
        if isinstance(other, EncryptedNumber):
            return self._add_encrypted(other)
        if isinstance(other, mpz):
            e = EncryptedNumber(self.pp, other)
            return self._add_encrypted(e)

    def __iadd__(self, other):
        if isinstance(other, EncryptedNumber):
            return self._add_encrypted(other)
        if isinstance(other, mpz):
            e = EncryptedNumber(self.pp, other)
            return self._add_encrypted(e)

    def __repr__(self):
        estr = self.ciphertext.digits()
        return "<EncryptedNumber {}...{}>".format(estr[:5], estr[-5:])

    def _add_encrypted(self, other):
        if self.pp != other.pp:
            raise ValueError(
                "Attempted to add numbers encrypted against " "different prameters!"
            )

        return EncryptedNumber(
            self.pp, self.ciphertext * other.ciphertext % self.pp.nsquare
        )

    def getrealsize(self):
        """
        returns the size of the ciphertext
        """
        return self.pp.bits * 2


"""
### **Full-Domain Hash**

This module computes a full domain hash value usign SHA256 hash function
"""

from Crypto.Hash import SHA256
from gmpy2 import gcd, mpz


class FDH(object):
    """
    The Full-Domain Hash scheme

    ## **Args**:
    -------------
    *bitsize* : `int` --
        The bitlength of the output of the FDH

    *N* : `int` --
        The modulus \\(N\\) such that the FDH output is in \\(\\mathbb{Z}^*_N\\)

    """

    def __init__(self, bitsize, N) -> None:
        super().__init__()
        self.bitsize = bitsize
        self.N = N

    def H(self, t):
        """
        Computes the FDH using SHA256

        It computes:
            $$\\textbf{SHA256}(x||0) ||\\textbf{SHA256}(x||1) || ... || \\textbf{SHA256}(x||c) \\mod N$$
        where \\(c\\) is a counter that keeps incrementing until the size of the output has *bitsize* length and the output falls in  \\(\\mathbb{Z}^*_N\\)

        ## **Args**:
        -------------
        *t* : `int` --
            The input of the hash function

        ## **Returns**:
        ----------------
        A value in \\(\\mathbb{Z}^*_N\\) of type `gmpy2.mpz`
        """
        counter = 1
        result = b""
        while True:
            while True:
                h = SHA256.new()
                h.update(
                    int(t).to_bytes(self.bitsize // 2, "big")
                    + counter.to_bytes(1, "big")
                )
                result += h.digest()
                counter += 1
                if len(result) < (self.bitsize // 8):
                    break
            r = mpz(int.from_bytes(result[-self.bitsize :], "big"))
            if gcd(r, self.N) == 1:
                break
            else:
                print("HAPPENED")
        return r
