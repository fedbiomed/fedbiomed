# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0
from typing import Union, Optional, List

from fedbiomed.common.logger import logger


class SecaggLomCrypter:
    """xxx"""

    def __init__(self) -> None:
        """Class constructor"""

        logger.debug("DUMMY PAYLOAD: SecaggLomCrypter initialization")

    def encrypt(
        self,
        num_nodes: int,
        current_round: int,
        params: List[float],
        temporary_key: str,
        clipping_range: Union[int, None] = None,
        weight: Optional[int] = None,
    ) -> List[int]:

        logger.debug(f"DUMMY PAYLOAD: SecaggLomCrypter does not encrypt, dummy key is {temporary_key}")
        return params

    def aggregate(
            self,
            current_round: int,
            num_nodes: int,
            params: List[List[int]],
            total_sample_size: int,
            clipping_range: Union[int, None] = None,
            num_expected_params: int = 1
    ) -> List[float]:

        logger.debug("DUMMY PAYLOAD: SecaggLomCrypter does not do secure aggregation")
        # dummy: just return params from one of the nodes ...
        return params[0]
