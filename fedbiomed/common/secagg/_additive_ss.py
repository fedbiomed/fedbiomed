import random
from math import log2

from fedbiomed.common.exceptions import FedbiomedSecaggCrypterError
from fedbiomed.common.constants import ErrorNumbers


class Secret:
    def __init__(self, secret):
        """
        Initializes the Secret class with the provided secret value.

        Args:
            secret (int | list[int]): The secret to be shared, either an integer or a list of integers.

        Raises:
            FedbiomedSecaggCrypterError: If the secret is not an int or a list of integers.
        """
        try:
            if isinstance(secret, int) or (isinstance(secret, list) and all(isinstance(i, int) for i in secret)):
                self.secret = secret
            else:
                raise ValueError("Secret must be an int or a list of int")
        except ValueError as exp:
            raise FedbiomedSecaggCrypterError(
                f"{ErrorNumbers.FB630}: Invalid secret format, {exp}"
            )

    def split(self, num_shares):
        """
        Splits the secret into the specified number of shares using additive secret sharing.
        The sum of the shares will equal the original secret.

        Args:
            num_shares (int): The number of shares to generate.

        Returns:
            list[Share]: A list of Share objects representing the split shares.

        Raises:
            FedbiomedSecaggCrypterError: If the secret type is neither an int nor a list of integers.
        """
        try:
            if isinstance(self.secret, int):
                shares = [random.randint(0, self.secret) for _ in range(num_shares - 1)]
                last_share = self.secret - sum(shares)
                shares.append(last_share)

            elif isinstance(self.secret, list):
                shares = []
                max_bit_length = max([value.bit_length() for value in self.secret])

                for value in self.secret:
                    partial_shares = [random.randint(0, (max_bit_length - int(log2(num_shares)))) for _ in range(num_shares - 1)]
                    partial_shares.append(value - sum(partial_shares))
                    shares.append(partial_shares)

                shares = list(map(list, zip(*shares)))

            else:
                raise ValueError("Secret must be an int or a list of int")
        
            return [Share(share) for share in shares]

        except ValueError as exp:
            raise FedbiomedSecaggCrypterError(
                f"{ErrorNumbers.FB630}: Failed to split the secret, {exp}"
            )
    @staticmethod
    def reconstruct(shares):
        """ 
        Given a list of shares, reconstructs the original secret.

        Args:
            shares (list[Share]): A list of Share objects to reconstruct the secret from.
        
        Returns:
            int | list[int]: The reconstructed secret, either an integer or a list of integers.

        Raises:
            FedbiomedSecaggCrypterError: If the shares are not of the same type (int or list).
        """

        try:
            if all(isinstance(share.value, int) for share in shares):
                return sum(share.value for share in shares)
            elif all(isinstance(share.value, list) for share in shares):
                return [sum(share.value[i] for share in shares) for i in range(len(shares[0].value))]
            else:
                raise ValueError("All shares must be of the same type (int or list of int).")
        
        except ValueError as exp:
            raise FedbiomedSecaggCrypterError(
                f"{ErrorNumbers.FB630}: Failed to reconstruct the secret, {exp}"
            )


class Share:
    def __init__(self, value):
        """
        Initializes the Share class with a given value.

        Args:
            value (int | list[int]): The value of the share, either an integer or a list of integers.

        Raises:
            FedbiomedSecaggCrypterError: If the value is neither an int nor a list of integers.
        """
        try:
            if isinstance(value, int) or (isinstance(value, list) and all(isinstance(i, int) for i in value)):
                self._value = value
            else:
                raise ValueError("Share value must be an int or a list of int")
        except ValueError as exp:
            raise FedbiomedSecaggCrypterError(
                f"{ErrorNumbers.FB630}: Invalid share value format, {exp}"
            )

    def __add__(self, other):
        """
        Adds two shares together. Supports both integer and list types.

        Args:
            other (Share | int | list[int]): The share or value to add.

        Returns:
            Share: A new Share object with the resulting sum.

        Raises:
            FedbiomedSecaggCrypterError: If the two values being added are not of the same type (both int or both list).
        """
        try:
            if isinstance(other, Share):
                other = other.value
            
            if isinstance(self.value, int) and isinstance(other, int):
                return Share(self.value + other)
            elif isinstance(self.value, list) and isinstance(other, list):
                return Share([x + y for x, y in zip(self.value, other)])
            else:
                raise TypeError("Both shares must be of the same type (int or list of int).")

        except (TypeError, ValueError) as exp:
            raise FedbiomedSecaggCrypterError(
                f"{ErrorNumbers.FB630}: Failed to add shares, {exp}"
            )

    def __repr__(self):
        """
        Returns the string representation of the Share object.

        Returns:
            str: The string representation of the Share value.
        """
        return f"Share({self.value})"
    
    @property
    def value(self):
        """
        Getter for the share's value.

        Returns:
            int | list[int]: The value of the share.
        """
        return self._value
    

shares = [Share([1, 2, 3]), Share([4, 5, 6]), Share([5, 13, 21])]
secret = [10, 20, 30]
reconstructed = Secret.reconstruct(shares)
print(reconstructed) # [10, 20, 30]