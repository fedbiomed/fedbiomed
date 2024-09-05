import random
from math import log2
from fedbiomed.common.exceptions import FedbiomedValueError, FedbiomedTypeError
from fedbiomed.common.constants import ErrorNumbers

class Secret:
    def __init__(self, secret):
        """
        Initializes the Secret class with the provided additive secret value.

        Args:
            secret (int | list[int]): The secret to be shared, either an integer or a list of integers.

        Raises:
            FedbiomedValueError: If the secret is not an int or a list of integers.
        """

        if not (isinstance(secret, int) or (isinstance(secret, list) and all(isinstance(i, int) for i in secret))):
            raise FedbiomedValueError("Secret must be an int or a list of int")
        self.secret = secret

    def split(self, num_shares, bit_length=None):
        """
        Splits the secret into the specified number of shares using additive secret sharing.
        The sum of the shares will equal the original secret.

        Args:
            num_shares (int): The number of shares to generate.
            bit_length (int, optional): The bit length of the shares. Defaults to None.

        Returns:
            list[Share]: A list of Share objects representing the split shares.

        Raises:
            FedbiomedValueError: If the number of shares is less than or equal to 0.
        """
        if num_shares <= 0:
            raise FedbiomedValueError("Number of shares must be greater than 0")
        if bit_length is not None and bit_length < int(log2(self.secret)):
            raise FedbiomedValueError("Bit length must be greater or equal than the secret's bit length")

        if isinstance(self.secret, int):
            bit_length = self.secret.bit_length() if bit_length is None else bit_length
            shares = [random.randint(0, 2 ** bit_length) for _ in range(num_shares - 1)]
            last_share = self.secret - sum(shares)
            shares.append(last_share)
        else:
            shares = []
            bit_length = max(self.secret) if bit_length is None else bit_length
            for value in self.secret:
                partial_shares = [random.randint(0, 2 ** bit_length) for _ in range(num_shares - 1)]
                partial_shares.append(value - sum(partial_shares))
                shares.append(partial_shares)
            shares = list(map(list, zip(*shares)))

        return [Share(share) for share in shares]

    @staticmethod
    def reconstruct(shares):
        """
        Given a list of shares, reconstructs the original secret by summing the shares together.

        Args:
            shares (list[Share]): A list of Share objects to reconstruct the secret from.

        Returns:
            int | list[int]: The reconstructed secret, either an integer or a list of integers.

        Raises:
            FedbiomedTypeError: If the shares are not of the same type (int or list).
        """
        if not all(isinstance(share.value, int) for share in shares) and not all(
                isinstance(share.value, list) for share in shares):
            raise FedbiomedTypeError("All shares must be of type Share")

        if all(isinstance(share.value, int) for share in shares):
            result = sum(share.value for share in shares)
        else:
            result = [sum(share.value[i] for share in shares) for i in range(len(shares[0].value))]

        return result


class Share:
    def __init__(self, value):
        """
        Initializes the Share class with a given value, representing an additive secret share.

        Args:
            value (int | list[int]): The value of the share, either an integer or a list of integers.

        Raises:
            FedbiomedTypeError: If the value is neither an int nor a list of integers.
        """
        if not (isinstance(value, int) or (isinstance(value, list) and all(isinstance(i, int) for i in value))):
            raise FedbiomedTypeError("Share value must be an int or a list of int")
        self._value = value

    def __add__(self, other):
        """
        Adds two shares together. Supports both integer and list types.

        Args:
            other (Share | int | list[int]): The share or value to add.

        Returns:
            Share: A new Share object with the resulting sum.

        Raises:
            FedbiomedTypeError: If the two values being added are not of the same type (both int or both list).
        """
        if isinstance(other, int):
            result = self.value + other
        elif isinstance(other, list):
            result = [self.value[i] + other[i] for i in range(len(self.value))]
        elif isinstance(other, Share):
            result = self.value + other.value
        else:
            raise FedbiomedTypeError("Share value must be an int or a list of int")
        return Share(result)

    def __repr__(self) -> str:
        return f"Share({self.value})"
    
    @staticmethod
    def add(shares_1, shares_2):
        """
        Adds two lists of shares together. Supports both integer and list types.

        Args:
            shares_1 (list[Share]): The first list of shares.
            shares_2 (list[Share]): The second list of shares.

        Returns:
            list[Share]: A list of Share objects with the resulting sums.

        Raises:
            FedbiomedTypeError: If the two lists of shares are not of the same length.
        """
        if len(shares_1) != len(shares_2):
            raise FedbiomedTypeError("Shares must be of the same length")
        # check if shares are of the same type
        if all(isinstance(share.value, int) for share in shares_1) != all(isinstance(share.value, int) for share in shares_2):
            raise FedbiomedTypeError("Shares must be of the same type")
        # sum if shares are integers
        if all(isinstance(share.value, int) for share in shares_1):
            result = [shares_1[i] + shares_2[i] for i in range(len(shares_1))]
        # sum if shares are lists
        else:
            result = [Share([shares_1[i].value[j] + shares_2[i].value[j] for j in range(len(shares_1[i].value))])
                      for i in range(len(shares_1))]
        return result

    @property
    def value(self):
        """
        Getter for the share's value.

        Returns:
            int | list[int]: The value of the share.
        """
        return self._value

secret = Secret(1024)

shares = secret.split(3, bit_length=10)
