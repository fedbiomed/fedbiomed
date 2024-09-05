import random
from math import log2

class Secret:
    def __init__(self, secret):
        """
        Initialize the Secret class with a given secret.
        The secret can be an integer or a list of integers.

        :param secret: The secret value to be shared (int or list of int).
        """
        # Check if the secret is either an int or a list of int
        if isinstance(secret, int) or (isinstance(secret, list) and all(isinstance(i, int) for i in secret)):
            self.secret = secret  # Assign the secret
        else:
            raise ValueError("Secret must be an int or a list of int")  # Raise an error if not valid

    def split(self, num_shares):
        """
        Split the secret into the given number of shares using additive secret sharing.
        The sum of all the shares will reconstruct the original secret.

        :param num_shares: Number of shares to generate.
        :return: List of Share objects.
        """
        # If the secret is an integer
        if isinstance(self.secret, int):
            # Generate num_shares - 1 random shares, and calculate the last share such that their sum equals the secret
            shares = [random.randint(0, self.secret) for _ in range(num_shares - 1)]
            last_share = self.secret - sum(shares)  # Last share ensures the sum equals the secret
            shares.append(last_share)

        # If the secret is a list of integers
        elif isinstance(self.secret, list):
            shares = []
            # Get the maximum bit length of the values in the list (used for bounded random values)
            max_bit_length = max([value.bit_length() for value in self.secret])

            for value in self.secret:
                # Create num_shares - 1 random partial shares with bounds based on bit length
                partial_shares = [random.randint(0, (max_bit_length - int(log2(num_shares)))) for _ in range(num_shares - 1)]
                partial_shares.append(value - sum(partial_shares))  # Ensure the sum equals the original value
                shares.append(partial_shares)  # Append the generated shares for each list element

            # Transpose the list so that each inner list corresponds to one party's shares
            shares = list(map(list, zip(*shares)))

        else:
            # Raise an error if the secret is not valid
            raise ValueError("Secret must be an int or a list of int")
        
        # Return each share wrapped as a Share object
        return [Share(share) for share in shares]


class Share:
    def __init__(self, value):
        """
        Initialize the Share class with a value. The value can be an integer or a list of integers.

        :param value: The share value (int or list of int).
        """
        # Check if the value is valid (either an int or list of int)
        if isinstance(value, int) or (isinstance(value, list) and all(isinstance(i, int) for i in value)):
            self.value = value  # Assign the share value
        else:
            raise ValueError("Share value must be an int or a list of int")  # Raise an error if not valid

    def __add__(self, other):
        """
        Add two shares together. The shares can be added if they are of the same type (int or list of int).

        :param other: Another Share object or a raw int/list to add.
        :return: A new Share object with the summed value.
        """
        # If other is a Share object, extract its value for addition
        if isinstance(other, Share):
            other = other.value
        
        # Handle addition of integers
        if isinstance(self.value, int) and isinstance(other, int):
            return Share(self.value + other)  # Return a new Share with summed integer values

        # Handle addition of lists of integers
        elif isinstance(self.value, list) and isinstance(other, list):
            # Element-wise addition of the two lists
            return Share([x + y for x, y in zip(self.value, other)])
        
        else:
            # Raise an error if the types don't match
            raise TypeError("Both shares must be of the same type (int or list of int).")

    def __repr__(self):
        """
        String representation of a Share object.
        :return: String format of the Share value.
        """
        return f"Share({self.value})"
    
    @property
    def value(self):
        return self.__value