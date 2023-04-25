# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

def matching_parties_servkey(context: dict, parties: list) -> bool:
    """Check if parties of given context are compatible with the parties
        of a secagg servkey element.

        Args:
            context: context to be compared with the secagg servkey element parties
            parties: the secagg servkey element parties

        Returns:
            True if this context can be used with this element, False if not.
        """
    # Need to ensure that:
    # - existing element was established for the same parties
    # - first party needs to be the same for both
    # - set of other parties needs to be the same for both (order can differ)
    #
    # eg: [ 'un', 'deux', 'trois' ] compatible with [ 'un', 'trois', 'deux' ]
    # but not with [ 'deux', 'un', 'trois' ]
    return (
        # Commented tests can be assumed from calling functions
        #
        # isinstance(context, dict) and
        # 'parties' in context and 
        # isinstance(context['parties'], list) and
        # len(context['parties']) >= 1 and
        # isinstance(parties, list) and
        parties[0] == context['parties'][0] and
        set(parties[1:]) == set(context['parties'][1:]))


def matching_parties_biprime(context: dict, parties: list) -> bool:
    """Check if parties of given context are compatible with the parties
        of a secagg biprime element.

        Args:
            context: context to be compared with the secagg biprime element parties
            parties: the secagg biprime element parties

        Returns:
            True if this context can be used with this element, False if not.
    """
    # Need to ensure that:
    # - either the existing element is not attached to specific parties (None)
    # - or existing element was established for the same parties or a superset of the parties
    return (
        # Commented tests can be assumed from calling functions
        #
        # isinstance(context, dict) and
        # 'parties' in context and 
        # isinstance(parties, list) and
        (
            context['parties'] is None or (
                # isinstance(context['parties'], list) and
                set(parties).issubset(set(context['parties']))
            )))
