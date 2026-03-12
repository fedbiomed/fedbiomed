# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Implementation of Federated Analytics Job class of the node component
"""

from typing import Any, Dict, Optional

from fedbiomed.common.constants import DatasetTypes, ErrorNumbers, FedbiomedError, Stats
from fedbiomed.common.dataset_types import DataReturnFormat
from fedbiomed.common.logger import logger
from fedbiomed.common.message import ErrorMessage, FAReply, FARequest
from fedbiomed.common.secagg import SecaggCrypter
from fedbiomed.node.dataset_manager import DatasetManager
from fedbiomed.node.secagg_manager import SecaggServkeyManager

from ._base_job import _BaseJob, _InternalJobError


class FAJob(_BaseJob):
    """
    This class represents the training part execute by a node in a given Federated Analytics job.
    """

    def __init__(
        self,
        root_dir: str,
        db: str,
        dataset_manager: DatasetManager,
        node_id: str,
        node_name: str,
        request: FARequest,
        allow_fa: bool,
    ) -> None:
        """Constructor of the class

        Args:
            root_dir: Root fedbiomed directory where node instance files will be stored.
            db: Path to the node's TinyDB database file (used to retrieve the
                secagg server-key context during encryption).
            dataset_manager: DatasetManager instance to retrieve datasets
            node_id: Node id
            node_name: Node name (Hospital name)
            request: FARequest message object containing all information about the FA task
            allow_fa: True if federated analytics is allowed on this node, False otherwise
        """
        super().__init__(root_dir, dataset_manager, node_id, node_name, request)

        self._db = db
        self._stats = request.stats
        self._dataset_id = request.dataset_id
        self._experiment_id = request.experiment_id
        self._fa_id = request.fa_id
        self._stats_args = request.stats_args
        self._dataset_schema = request.dataset_schema
        self._allow_fa = allow_fa
        self._secagg = request.secagg if hasattr(request, "secagg") else False
        self._secagg_arguments = (
            request.secagg_arguments if hasattr(request, "secagg_arguments") else {}
        )

    def _get_encryption_params(self) -> Optional[Dict]:
        """Retrieve encryption parameters from the local secagg database.

        Mirrors the key-lookup used by _JLSRound: the server-key share
        (key + biprime) is fetched from the node's TinyDB using the
        secagg_servkey_id sent in the FA request, not read from the request
        arguments directly (which would expose keys over the wire).

        Returns:
            Dict with num_nodes, key, biprime, clipping_range, current_round,
            or None if secagg is disabled or the context cannot be found.
        """
        if not self._secagg or not self._secagg_arguments:
            return None

        secagg_servkey_id = self._secagg_arguments.get("secagg_servkey_id")
        if not secagg_servkey_id:
            logger.warning(
                "Secagg enabled but 'secagg_servkey_id' missing in secagg_arguments, "
                "skipping encryption"
            )
            return None

        try:
            manager = SecaggServkeyManager(self._db)
            secagg_servkey = manager.get(
                secagg_id=secagg_servkey_id,
                experiment_id=self._experiment_id,
            )
        except Exception as e:
            logger.warning(f"Failed to retrieve secagg server key: {e}")
            return None

        if secagg_servkey is None:
            logger.warning(
                f"Secagg server key '{secagg_servkey_id}' not found in database, "
                "skipping encryption"
            )
            return None

        # num_nodes is the number of encrypting nodes (researcher excluded).
        # The researcher sets this explicitly so JLS encode/decode use the same
        # add_ops value (see fix for Issue 2 / n_users mismatch).
        num_nodes = self._secagg_arguments.get(
            "num_nodes", len(self._secagg_arguments.get("parties", [])) - 1
        )

        return {
            "num_nodes": num_nodes,
            "key": secagg_servkey["context"]["server_key"],
            "biprime": secagg_servkey["context"]["biprime"],
            "clipping_range": self._secagg_arguments.get("secagg_clipping_range"),
            "current_round": 1,
        }

    def _encrypt_output(self, output: Dict) -> Dict:
        """Encrypt output statistics for secure aggregation.

        For histograms, only the counts are encrypted (bin_edges remain in clear).
        For ``mean``, when a numeric ``count`` sibling is present in the same
        dict, the value ``sum = mean * count`` is encrypted instead of ``mean``
        directly; the ``_sum_encoded`` marker is set so the researcher can
        recover the weighted global mean as ``total_sum / total_count``.
        All other numeric values are encrypted as-is.

        Args:
            output: The computed statistics output.

        Returns:
            Encrypted output dictionary.
        """
        enc_params = self._get_encryption_params()
        if enc_params is None:
            return output

        crypter = SecaggCrypter()

        def _encrypt_scalar(scalar: float) -> dict:
            encrypted = crypter.encrypt(
                num_nodes=enc_params["num_nodes"],
                current_round=enc_params["current_round"],
                params=[scalar],
                key=enc_params["key"],
                biprime=enc_params["biprime"],
                clipping_range=enc_params["clipping_range"],
            )
            return {"_encrypted": True, "value": encrypted[0]}

        def encrypt_value(val: Any) -> Any:
            if isinstance(val, dict):
                # When both 'mean' and 'count' are numeric siblings, encrypt
                # sum = mean * count instead of mean.  This lets the researcher
                # recover the true weighted global mean (total_sum / total_count)
                # rather than the arithmetic average of per-node means.
                mean_v = val.get("mean")
                count_v = val.get("count")
                encode_mean_as_sum = isinstance(mean_v, (int, float)) and isinstance(
                    count_v, (int, float)
                )
                result = {}
                for k, v in val.items():
                    if k == "histogram":
                        result[k] = self._encrypt_histogram(v, enc_params, crypter)
                    elif k == "bin_edges":
                        result[k] = v
                    elif k == "mean" and encode_mean_as_sum:
                        try:
                            enc = _encrypt_scalar(float(v) * float(count_v))
                            enc["_sum_encoded"] = True
                            result[k] = enc
                        except Exception as e:
                            logger.warning(f"Failed to encrypt mean as sum: {e}")
                            result[k] = v
                    else:
                        result[k] = encrypt_value(v)
                return result
            elif isinstance(val, list):
                return [encrypt_value(item) for item in val]
            elif isinstance(val, (int, float)):
                try:
                    return _encrypt_scalar(float(val))
                except Exception as e:
                    logger.warning(f"Failed to encrypt value {val}: {e}")
                    return val
            return val

        return encrypt_value(output)

    @staticmethod
    def _encrypt_histogram(
        histogram: Dict, enc_params: Dict, crypter: SecaggCrypter
    ) -> Dict:
        """Encrypt histogram counts while keeping bin_edges in clear.

        Args:
            histogram: Histogram dict with 'bin_edges' and 'counts'.
            enc_params: Encryption parameters from _get_encryption_params().
            crypter: Pre-constructed SecaggCrypter instance.

        Returns:
            Histogram with encrypted counts.
        """
        if not isinstance(histogram, dict):
            return histogram

        result = {"bin_edges": histogram.get("bin_edges")}

        counts = histogram.get("counts", [])
        if counts:
            try:
                encrypted_counts = crypter.encrypt(
                    num_nodes=enc_params["num_nodes"],
                    current_round=enc_params["current_round"],
                    params=[float(c) for c in counts],
                    key=enc_params["key"],
                    biprime=enc_params["biprime"],
                    clipping_range=enc_params["clipping_range"],
                )
                result["counts"] = [
                    {"_encrypted": True, "value": c} for c in encrypted_counts
                ]
            except Exception as e:
                logger.warning(f"Failed to encrypt histogram counts: {e}")
                result["counts"] = counts

        return result

    def _build_args_for_dataset(self, dataset_entry: dict) -> dict:
        """Generate dataset arguments based on the dataset type by default."""
        data_type = dataset_entry.get("data_type")
        dataset_type = DatasetTypes.get_type_by_value(data_type)

        match dataset_type:
            case None:
                # This should not happen, but this check is added for safety
                raise FedbiomedError(
                    f"Dataset entry contains unsupported dataset type '{data_type}'."
                )
            case DatasetTypes.TABULAR:
                # Take keys in 'dtypes' and pass them as ``input_columns`` to the dataset constructor
                return {"input_columns": list(dataset_entry.get("dtypes", {}))}
            case DatasetTypes.IMAGES | DatasetTypes.DEFAULT | DatasetTypes.MEDNIST:
                # For image datasets, no dataset arguments are passed by default
                return {}
            case DatasetTypes.MEDICAL_FOLDER:
                # Take keys in 'shape' and pass them as ``data_modalities`` to the dataset constructor
                return {"data_modalities": list(dataset_entry.get("shape", {}))}
            case _:
                raise FedbiomedError(
                    f"Dataset arguments by default are not implemented for dataset type '{data_type}'."
                )

    def run(self) -> FAReply | ErrorMessage:
        """Run FA job and return FAReply message or ErrorMessage in case of failure."""

        if not self._allow_fa:
            return self._build_error_msg(
                "Federated Analytics are not allowed on this node by node configuration.",
                errnum=ErrorNumbers.FB325.value,
            )

        # Validate that at least one of stats or stats_args is provided
        if self._stats is None and not self._stats_args:
            return self._build_error_msg(
                msg="At least one of 'stats' or 'stats_args' must be provided.",
                errnum=ErrorNumbers.FB325.value,
            )

        # Validate that all requested stats and stats_args keys are valid enum values
        valid_stats = {s.value for s in Stats}
        if self._stats is not None:
            invalid = [s for s in self._stats if s not in valid_stats]
            if invalid:
                return self._build_error_msg(
                    msg=f"'stats' contains unsupported values: {invalid}",
                    errnum=ErrorNumbers.FB325.value,
                )
        if self._stats_args is not None:
            invalid = [k for k in self._stats_args if k not in valid_stats]
            if invalid:
                return self._build_error_msg(
                    msg=f"'stats_args' contains unsupported keys: {invalid}",
                    errnum=ErrorNumbers.FB325.value,
                )

        try:
            dataset = self._build_dataset(DataReturnFormat.SKLEARN)
        except _InternalJobError as e:
            return self._build_error_msg(msg=repr(e), errnum=ErrorNumbers.FB325.value)

        # Use compute_stats to handle all statistics
        if not hasattr(dataset, "compute_stats"):
            return self._build_error_msg(
                msg="Dataset does not support analytics method 'compute_stats'.",
                errnum=ErrorNumbers.FB325.value,
            )

        logger.debug(
            f"FA Request received and database built, executing analytics: compute_stats (requested: {self._stats})"
        )

        # Prepare kwargs
        kwargs = {
            "stats": self._stats,
            "stats_args": self._stats_args,
            "dataset_schema": self._dataset_schema,
        }

        try:
            output: Dict = dataset.compute_stats(**kwargs)
            logger.debug(f"Analytics executed, output: {output}")
        except Exception as e:
            return self._build_error_msg(
                msg=(
                    f"Error during execution of analytics '{self._stats}' "
                    f"on node='{self._node_id}': {repr(e)}"
                ),
                errnum=ErrorNumbers.FB325.value,
            )

        if self._secagg:
            output = self._encrypt_output(output)

        return FAReply(
            request_id=self._request_id,
            researcher_id=self._researcher_id,
            experiment_id=self._experiment_id,
            fa_id=self._fa_id,
            stats=self._stats,
            node_id=self._node_id,
            node_name=self._node_name,
            output=output,
        )
