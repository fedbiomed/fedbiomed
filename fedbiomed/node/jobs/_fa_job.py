# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Implementation of Federated Analytics Job class of the node component
"""

from typing import Dict, Optional

import polars as pl

from fedbiomed.common.constants import (
    DatasetTypes,
    ErrorNumbers,
    FedbiomedError,
    SAParameters,
    Stats,
)
from fedbiomed.common.dataloadingplan import DataLoadingPlan
from fedbiomed.common.dataset import REGISTRY_CONTROLLERS, Dataset
from fedbiomed.common.dataset_types import DataReturnFormat
from fedbiomed.common.exceptions import FedbiomedSecureAggregationError
from fedbiomed.common.logger import logger
from fedbiomed.common.message import ErrorMessage, FAReply, FARequest
from fedbiomed.common.utils import flatten_fa_output
from fedbiomed.node.dataset_manager import DatasetManager
from fedbiomed.node.secagg import SecaggRound

from ._base_job import _BaseJob, _InternalJobError


class FAJob(_BaseJob):
    """
    Represents the analytics execution performed by a node in a Federated Analytics job.
    """

    def __init__(
        self,
        root_dir: str,
        dataset_manager: DatasetManager,
        node_id: str,
        node_name: str,
        request: FARequest,
        allow_fa: bool,
        db: str = "",
        secagg_active: bool = False,
        force_secagg: bool = False,
        secagg_arguments: Optional[Dict] = None,
    ) -> None:
        """Constructor of the class

        Args:
            root_dir: Root fedbiomed directory where node instance files will be stored.
            dataset_manager: DatasetManager instance to retrieve datasets
            node_id: Node id
            node_name: Node name (Hospital name)
            request: FARequest message object containing all information about the FA task
            allow_fa: True if federated analytics is allowed on this node, False otherwise
            db: Path to the node database file; required when secagg is active.
            secagg_active: True if secure aggregation is enabled in node config.
            force_secagg: True if the node mandates secure aggregation; an FA request
                without secagg_arguments is then rejected.
            secagg_arguments: Secure aggregation arguments forwarded from the FARequest;
                None means the researcher did not request encryption.
        """
        super().__init__(root_dir, dataset_manager, node_id, node_name, request)

        self._stats = request.stats
        self._dataset_id = request.dataset_id
        self._experiment_id = request.experiment_id
        self._fa_id = request.fa_id
        self._stats_args = request.stats_args
        self._dataset_schema = request.dataset_schema
        self._allow_fa = allow_fa
        self._db = db
        self._secagg_active = secagg_active
        self._force_secagg = force_secagg
        self._secagg_arguments = secagg_arguments

    def _build_args_for_dataset(self, dataset_entry: dict) -> dict:
        """Extract dataset constructor arguments from a dataset registry entry.

        Reads the ``data_type`` field of ``dataset_entry`` and returns a kwargs
        dict suitable for passing to the corresponding dataset class constructor.

        Args:
            dataset_entry: Dataset metadata dict as stored in the node dataset
                table. Expected keys vary by data type.

        Returns:
            A kwargs dict for the dataset class constructor, e.g.
            ``{"input_columns": ["age", "weight"]}`` for tabular data.

        Raises:
            _InternalJobError: if ``data_type`` is unknown or not yet supported.
        """
        data_type = dataset_entry.get("data_type")
        dataset_type = DatasetTypes.get_type_by_value(data_type)

        match dataset_type:
            case None:
                # This should not happen, but this check is added for safety
                raise _InternalJobError(
                    f"Dataset entry contains unsupported dataset type '{data_type}'."
                )
            case DatasetTypes.TABULAR:
                # Keep only columns whose dtype produces a numerical numpy array via to_numpy()
                return {
                    "input_columns": [
                        col
                        for col, dtype_name in dataset_entry.get("dtypes", {}).items()
                        if (cls := getattr(pl, dtype_name, None)) is not None
                        and cls().is_numeric()
                    ]
                }
            case DatasetTypes.IMAGES | DatasetTypes.DEFAULT | DatasetTypes.MEDNIST:
                # For image datasets, no dataset arguments are passed by default
                return {}
            case DatasetTypes.MEDICAL_FOLDER:
                # Take keys in 'shape' and pass them as ``data_modalities`` to the dataset constructor
                return {"data_modalities": list(dataset_entry.get("shape", {}))}
            case _:
                raise _InternalJobError(
                    f"Dataset arguments by default are not implemented for dataset type '{data_type}'."
                )

    def _build_dataset(
        self,
        return_type: DataReturnFormat = DataReturnFormat.SKLEARN,
    ) -> Dataset:
        """Build a ready-to-use dataset instance from ``self._dataset_id``.

        Args:
            return_type: Output format for the data loader.

        Returns:
            Fully initialised ``Dataset`` instance.

        Raises:
            _InternalJobError: if the dataset entry cannot be found, its type is
                unsupported, the DLP cannot be deserialised, or initialisation fails.
        """
        # recover dataset entry
        dataset_entry, _ = self._dataset_manager.get_dataset_entry_by_id(
            self._dataset_id
        )
        if dataset_entry is None:
            raise _InternalJobError(
                f"Cannot find requested dataset in local datasets: dataset_id='{self._dataset_id}' "
                f"on node='{self._node_id}'"
            )

        # check that data type is supported and get dataset class
        data_type = dataset_entry.get("data_type")
        dataset_type = DatasetTypes.get_type_by_value(data_type)
        if dataset_type not in REGISTRY_CONTROLLERS:
            raise _InternalJobError(
                f"Data type '{dataset_type}' not supported in jobs, available types: "
                f"{list(REGISTRY_CONTROLLERS.keys())}"
            )

        # get controller parameters
        # `root` is in both `dataset_parameters` and `path`; they must be the same.
        controller_kwargs = {
            **dataset_entry.get("dataset_parameters", {}),
            "root": dataset_entry.get("path"),
        }

        # recover dlp if any
        if "dlp_id" in dataset_entry:
            dlp_metadata = self._dataset_manager.get_dlp_by_id(dataset_entry["dlp_id"])
            try:
                controller_kwargs["dlp"] = DataLoadingPlan().deserialize(*dlp_metadata)
            except FedbiomedError as e:
                raise _InternalJobError(
                    f"Cannot recover dlp on node={self._node_id}: {repr(e)}"
                ) from e

        # REGISTRY_CONTROLLERS maps dataset_type -> (controller_cls, dataset_cls)
        _, dataset_cls = REGISTRY_CONTROLLERS[dataset_type]

        try:
            # build with type-specific args then attach controller config and output format
            dataset = dataset_cls(**self._build_args_for_dataset(dataset_entry))
            dataset.load(to_format=return_type, **controller_kwargs)
        except FedbiomedError as e:
            raise _InternalJobError(
                f"Cannot initialize dataset on node='{self._node_id}': {repr(e)}"
            ) from e

        try:
            self._dataset_manager.validate_samples(len(dataset))
        except FedbiomedError as e:
            raise _InternalJobError(
                f"Dataset '{self._dataset_id}' does not meet minimum sample requirement: {repr(e)}"
            ) from e

        return dataset

    def _check_clipping_overflow(
        self, flat: list, schema: list, clip: float
    ) -> Optional[ErrorMessage]:
        """Reject statistics outside [-clip, clip]; encrypting them corrupts the aggregate.

        Args:
            flat: Flattened computed analytics values.
            schema: Per-value key-paths (parallel to `flat`); empty path = bare scalar.
            clip: Symmetric clipping bound.

        Returns:
            An ErrorMessage naming the offenders, or None if all values fit.
        """
        named = []
        n_over = 0
        for value, key_path in zip(flat, schema, strict=True):
            if value > clip or value < -clip:
                n_over += 1
                path = ".".join(str(k) for k in key_path)
                if path:  # bare scalars have no key-path → no name to report
                    named.append(path)

        if not n_over:
            return None

        where = f" Offending statistic(s): {', '.join(named)}." if named else ""
        return self._build_error_msg(
            msg=(
                f"{n_over} computed analytics value(s) exceed the secure "
                f"aggregation clipping range; encrypting would "
                f"corrupt the result.{where} Restrict the request."
            ),
            errnum=ErrorNumbers.FB325.value,
        )

    def run(self) -> FAReply | ErrorMessage:
        """Run FA job and return FAReply message or ErrorMessage in case of failure."""

        if not self._allow_fa:
            return self._build_error_msg(
                "Federated Analytics are not allowed on this node by node configuration.",
                errnum=ErrorNumbers.FB325.value,
            )

        # Validate that all requested stats are valid enum values
        valid_stats = {s.value for s in Stats}
        if self._stats is not None:
            invalid = [s for s in self._stats if s not in valid_stats]
            if invalid:
                return self._build_error_msg(
                    msg=f"'stats' contains unsupported values: {invalid}",
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

        try:
            secagg_round = SecaggRound(
                db=self._db,
                node_id=self._node_id,
                secagg_arguments=self._secagg_arguments,
                secagg_active=self._secagg_active,
                force_secagg=self._force_secagg,
                experiment_id=self._experiment_id,
            )
        except FedbiomedSecureAggregationError as e:
            return self._build_error_msg(msg=repr(e), errnum=ErrorNumbers.FB325.value)

        if secagg_round.use_secagg:
            # FA uses fixed ranges, never recovered from the request.
            flat, schema = flatten_fa_output(output)
            clip_error = self._check_clipping_overflow(
                flat, schema, clip=SAParameters.FA_CLIPPING_RANGE
            )
            if clip_error is not None:
                return clip_error
            encrypted_params = secagg_round.scheme.encrypt(
                flat,
                current_round=self._secagg_arguments.get("fa_round", 1),
                weight=1,
                target_range=SAParameters.FA_TARGET_RANGE,
                clipping_range=SAParameters.FA_CLIPPING_RANGE,
            )

            return FAReply(
                request_id=self._request_id,
                researcher_id=self._researcher_id,
                experiment_id=self._experiment_id,
                fa_id=self._fa_id,
                stats=self._stats,
                node_id=self._node_id,
                node_name=self._node_name,
                encrypted=True,
                params_encrypted=encrypted_params,
                output_schema=schema,
            )

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
