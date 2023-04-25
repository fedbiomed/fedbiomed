# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import uuid
import os
import subprocess
from typing import Tuple

from fedbiomed.common.exceptions import FedbiomedMPCControllerError
from fedbiomed.common.utils import ROOT_DIR
from fedbiomed.common.logger import logger
from fedbiomed.common.constants import ErrorNumbers, ComponentType


class MPCController:

    def __init__(
            self,
            tmp_dir: str,
            component_type: ComponentType,
            component_id: str,
    ) -> None:
        """Multi Party Computation for negotiating cryptographic material with other parties.

        Args:
            tmp_dir: directory use as basedir for temporary files 
            component_type: type of component (researcher or node) this 
            component_id: unique ID of this component

        Raises:
            FedbiomedMPCControllerError: cannot create directory for temporary files or MPC files
        """

        # Get root directory of fedbiomed
        self._root = ROOT_DIR
        self._component_type = component_type
        mpc_controller_id = str(uuid.uuid4())

        # Use tmp dir to write files
        self._tmp_dir = os.path.join(tmp_dir, 'MPC', component_id, mpc_controller_id)

        # Create TMP dir for MPC logs if it is not existing
        if not os.path.isdir(self._tmp_dir):
            try:
                os.makedirs(self._tmp_dir)
            except Exception as e:
                raise FedbiomedMPCControllerError(
                    f"{ErrorNumbers.FB620.value}: Cannot create directory for MPC temporary files : {e}"
                )

        self._mpc_script = os.path.join(self._root, 'scripts', 'fedbiomed_mpc')
        self._mpc_dir = os.path.join(self._tmp_dir, 'MP-SPDZ')
        self._mpc_data_dir = os.path.join(self._mpc_dir , 'Player-Data')

        if not os.path.isdir(self._mpc_data_dir):
            try:
                os.makedirs(self._mpc_data_dir)
            except Exception as e:
                raise FedbiomedMPCControllerError(
                    f"{ErrorNumbers.FB620.value}: Cannot create directory for MPC config data : {e}"
                )

    @property
    def mpc_data_dir(self) -> str:
        """Getter for MPC config data directory

        Returns:
            directory for MPC config data directory
        """
        return self._mpc_data_dir

    @property
    def tmp_dir(self) -> str:
        """Getter for MPC temporary files directory

        Returns:
            directory for MPC temporary files directory
        """
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

        Raises:
            FedbiomedMPCControllerError: MPC computation error, or bad parameters
        """

        output_file = os.path.join(self._tmp_dir, "Output")
        input_file = os.path.join(self._tmp_dir, "Input")

        i_f_command = ["-if", input_file] if party_number != 0 else []
        o_f_command = ["-of", output_file] if party_number == 0 else []

        command = [self._component_type.name.lower(),
                   self._mpc_dir,
                   "shamir-server-key",
                   "-pn", str(party_number),
                   "-nop", str(num_parties),
                   *i_f_command, *o_f_command,
                   "-aip", ip_addresses, "--compile"]
        status, output = self._exec(command=command)

        if not status:
            error_message = f"{ErrorNumbers.FB620.value}: MPC computation for {self._component_type.name} " \
                "is not successful."
            logger.debug(f"{error_message}. Details: {output}")
            raise FedbiomedMPCControllerError(error_message)

        return f"{output_file}-P{party_number}-0" if party_number == 0 else \
            f"{input_file}-P{party_number}-0"

    def _exec(
            self,
            command: list
    ) -> Tuple[bool, str]:
        """Execute the MPC script

        Args:
            command: command and parameters for the MPC script

        Returns:
            a tuple composed of a boolean (True if command successfully executed) and
                a string (the output of the command)

        Raises:
            FedbiomedMPCControllerError: command execution error, or bad output
        """

        try:
            process = subprocess.Popen([self._mpc_script, *command],
                                       stderr=subprocess.STDOUT,
                                       stdout=subprocess.PIPE)
            process.wait()
            status = True if process.returncode == 0 else False
            output, _ = process.communicate()
            logger.debug("MPC protocol output: " + f"\n {output.decode('utf-8')}".replace('\n', '\n\t\t\t\t\t\t'))
        except Exception as e:
            logger.debug(f"{ErrorNumbers.FB620.value} MPC protocol error {e}")
            raise FedbiomedMPCControllerError(
                f"{ErrorNumbers.FB620.value}: Unexpected error while executing MPC protocol. {e}"
            )

        return status, output.decode('utf-8')
