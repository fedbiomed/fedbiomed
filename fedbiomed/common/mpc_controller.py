import os
import subprocess
from typing import Union

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
        root = get_fedbiomed_root()

        self._mpc_script = os.path.join(root, 'scripts', 'fedbiomed_mpc')
        self._mpc_data_dir = os.path.join(root, 'modules', 'MP-SPDZ', 'Player-Data')

        if not os.path.isdir(self._mpc_data_dir):
            os.makedirs(self._mpc_data_dir)

        # Use tmp dir to write files
        self.tmp_dir = os.path.join(tmp_dir, 'MPC', component_id)

        # Create TMP dir for MPC logs if it is not existing
        if not os.path.isdir(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    @property
    def mpc_data_dir(self):
        return self._mpc_data_dir

    def exec_shamir(
            self,
            party_number: int,
            num_parties: int,
            ip_addresses: str,
    ) -> str:
        """

        """

        output_file = os.path.join(self.tmp_dir, f"Output")
        input_file = os.path.join(self.tmp_dir, f"Input")

        if party_number == 0 and output_file is None:
            raise FedbiomedMPCControllerError(
                f"Party 0 (aggregator) should have input input and output file defined"
            )

        if party_number != 0 and input_file is None:
            raise FedbiomedMPCControllerError(
                f"Nodes should have output file defined for multi party computation"
            )

        i_f_command = ["-if",  input_file] if input_file is not None else []
        o_f_command = ["-of", output_file] if output_file is not None else []

        command = ["shamir-server-key",
                   "-pn", str(party_number),
                   "-nop", str(num_parties),
                   *i_f_command, *o_f_command,
                   "-aip", ip_addresses, "--compile"]

        if not self._exec(command=command):
            raise FedbiomedMPCControllerError(
                f"{ErrorNumbers.FB620.value} MPC computation for is not successful."
            )

        return f"{output_file}-P{party_number}-0" if party_number == 0 else \
            f"{input_file}-P{party_number}-0"

    def _exec(
            self,
            command: list
    ) -> bool:

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
                f"{ErrorNumbers.FB620.value}: Unexpected error while executing MPC protocol"
            )

        return status
