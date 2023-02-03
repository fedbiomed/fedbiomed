# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess

from fedbiomed.common.exceptions import FedbiomedMPCControllerError
from fedbiomed.common.utils import get_fedbiomed_root
from fedbiomed.common.logger import logger
from fedbiomed.common.constants import ErrorNumbers


class MPCController:

    def __init__(
            self,
            tmp_dir: str,
            component_id: str,
    ) -> None:
        """

        Args:
            tmp_dir:
            component_id:

        """

        # Get root directory of fedbiomed
        self._root = get_fedbiomed_root()

        self._mpc_script = os.path.join(self._root, 'scripts', 'fedbiomed_mpc')
        self._mpc_data_dir = os.path.join(self._root, 'modules', 'MP-SPDZ', 'Player-Data')

        if not os.path.isdir(self._mpc_data_dir):
            os.makedirs(self._mpc_data_dir)

        # Use tmp dir to write files
        self._tmp_dir = os.path.join(tmp_dir, 'MPC', component_id)

        # Create TMP dir for MPC logs if it is not existing
        if not os.path.isdir(self._tmp_dir):
            os.makedirs(self._tmp_dir)

    @property
    def mpc_data_dir(self):
        return self._mpc_data_dir

    @property
    def tmp_dir(self):
        return self._tmp_dir

    def exec_shamir(
            self,
            party_number: int,
            num_parties: int,
            ip_addresses: str,
    ) -> str:
        """Executes shamir protocol

        Args:
            party_number: The party number whose executing the protocol
            num_parties: NUm of total parties participating to multi party computation
            ip_addresses: The file path where IP addresses of the parties. This file is supposed
                to respect the order of parties.

        Returns:
            Path to the file where the input value (key-share) of the parties or output value of the server(server key)
                is written by the protocol.
        """

        if not isinstance(num_parties, int) or num_parties < 3:
            raise FedbiomedMPCControllerError(
                f"{ErrorNumbers.FB620}. Number of parties should be at least 3 but got "
                f"{type(num_parties)} {num_parties}"
            )

        output_file = os.path.join(self._tmp_dir, f"Output")
        input_file = os.path.join(self._tmp_dir, f"Input")

        i_f_command = ["-if", input_file] if party_number != 0 else []
        o_f_command = ["-of", output_file] if party_number == 0 else []

        command = ["shamir-server-key",
                   "-pn", str(party_number),
                   "-nop", str(num_parties),
                   *i_f_command, *o_f_command,
                   "-aip", ip_addresses, "--compile"]
        status, output = self._exec(command=command)

        if not status:
            error_message = f"{ErrorNumbers.FB620.value}: MPC computation for is not successful."
            logger.debug(f"{error_message}. Details: {output}")
            raise FedbiomedMPCControllerError(error_message)

        return f"{output_file}-P{party_number}-0" if party_number == 0 else \
            f"{input_file}-P{party_number}-0"

    def _exec(
            self,
            command: list
    ) -> bool:

        if not isinstance(command, list):
            raise FedbiomedMPCControllerError(
                f"{ErrorNumbers.FB620} Command should be presented as list where "
                f"each element is an option or value")

        try:
            process = subprocess.Popen([self._mpc_script, *command],
                                       stderr=subprocess.STDOUT,
                                       stdout=subprocess.PIPE)
            process.wait()
            status = True if process.returncode == 0 else False
            output, _ = process.communicate()
            logger.debug(f"MPC protocol output: {output}")
        except Exception as e:
            logger.debug(f"{ErrorNumbers.FB620.value} MPC protocol error {e}")
            raise FedbiomedMPCControllerError(
                f"{ErrorNumbers.FB620.value}: Unexpected error while executing MPC protocol. {e}"
            )

        return status, output
