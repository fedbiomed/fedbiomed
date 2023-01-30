import os
import subprocess
from typing import Union

from fedbiomed.common.exceptions import FedbiomedMPCControllerError
from fedbiomed.common.utils import get_fedbiomed_root


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

        self._mpc_script = os.path.join(
            root, 'scripts', 'fedbiomed_mpc'
        )

        self._mpc_data = os.path.join(
            root, 'modules', 'MP-SPDZ', 'Player-Data'
        )

        # Use tmp dir to write files
        self.tmp_dir = os.path.join(tmp_dir, 'MPC', component_id)

        # Create TMP dir for MPC logs if it is not existing
        if not os.path.isdir(self.tmp_dir):
            os.makedirs(self.tmp_dir)


    def exec_shamir(
            self,
            party_number: int,
            num_parties: int,
            input_file: Union[str, None] = None,
            output_file: Union[str, None] = None,

    ):

        prefix = f"P{party_number}"

        if party_number == 0 and output_file is None:
            raise FedbiomedMPCControllerError(
                f"Party 0 (aggregator) should have input input and output file defined"
            )

        if party_number != 0 and input_file is None:
            raise FedbiomedMPCControllerError(
                f"Nodes input input file defined for multi party computation"
            )

        input_file = f"-if {input_file}" if input_file is not None else ""
        output_file = f"-of {output_file}" if output_file is not None else ""

        command = f"shamir-server-key -pn {party_number} -np {num_parties} {output_file} {input_file} -c"

    def exec(
            self,
            command,
            ip_addresses: str
    ) -> None:

        try:
            sts = subprocess.Popen(f"{self._mpc_script} {command} -aip {ip_addresses}", shell=True).wait()
        except Exception as e:
            raise FedbiomedMPCControllerError(f"Error while executing MPC command: {e}")

        pass
