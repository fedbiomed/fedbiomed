"""Class wrapping the training routine of a node at a given round."""

import inspect
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import declearn

from fedbiomed.common.constants import ErrorNumbers, TrainingPlanApprovalStatus
from fedbiomed.common.data import (
    DataLoadingPlan,
    DataManager,
    TorchDataManager,
    TypeDataManager,
)
from fedbiomed.common.exceptions import (
    FedbiomedError,
    FedbiomedRoundError,
    FedbiomedTrainingPlanError,
)
from fedbiomed.common.history_monitor import HistoryMonitor
from fedbiomed.common.logger import logger
from fedbiomed.common.message import NodeMessages, TrainReply
from fedbiomed.common.repository import Repository
from fedbiomed.common.training_args import TrainingArgs
from fedbiomed.common.training_plans import TrainingPlan
from fedbiomed.node.environ import environ
from fedbiomed.node.training_plan_security_manager import (
    TrainingPlanSecurityManager,
)


class Round:
    """Class wrapping the training routine of a node at a given round."""

    def __init__(
        self,
        model_kwargs: Dict[str, Any],
        training_kwargs: Dict[str, Any],
        dataset: Dict[str, Any],
        training_plan_url: str,
        params_url: str,
        aux_vars_url: Optional[str] = None,
        training: bool = True,
        job_id: str = "",
        researcher_id: str = "",
        history_monitor: Optional[HistoryMonitor] = None,
        node_args: Optional[Dict[str, Any]] = None,
        dlp_and_loading_block_metadata: Optional[
            Tuple[Dict[str, Any], List[Dict[str, Any]]]
        ] = None,
    ) -> None:
        """Constructor of the class

        Args:
            model_kwargs: Keyword arguments to parametrize the model.
                Passed as `model_args` to `training_plan.post_init`.
            training_kwargs: Keyword arguments to parametrize training, but
                also evaluation and data loading. Parsed using `TrainingArgs`.
            dataset: Metadata about the local dataset, enabling its proper use.
                Keys: "dataset_id", "path" and optionally "dataset_parameters"
                (the latter being compatible with some specific datasets only).
            training_plan_url: URL from which to download the training plan
                JSON dump file.
            params_url: URL from which to download and at which to upload model
                parameters, before and after training, as a JSON dump file.
            aux_vars_url: URL from which to download and at which to upload
                auxiliary variables, before and after training, as a JSON dump file.
            training: Whether to conduct training or not.
            job_id: Job identifier string.
            researcher_id: Researcher identifier string.
            history_monitor: HistoryMonitor instance that collects real-time
                feed-back to the end-user during training and evaluation.
            node_args: Keyword arguments collected from the node's commandline.
                Passed as `node_args` to `training_plan.training_routine`.
            dlp_and_loading_block_metadata: Optional parameters to add a
                `DataLoadingPlan` to the dataset specified by the training
                plan. Only used when a `TorchDataManager` handles the data
                and said data is a `torch.utils.data.Dataset` subclass that
                implements a `set_dlp` method.
        """
        self.dataset = dataset
        self.training_plan_url = training_plan_url
        self.params_url = params_url
        self.aux_vars_url = aux_vars_url
        self.job_id = job_id
        self.researcher_id = researcher_id
        self.history_monitor = history_monitor
        self.node_args = node_args
        self.repository = Repository(
            environ["UPLOADS_URL"], environ["TMP_DIR"], environ["CACHE_DIR"]
        )
        self.training = training
        self._dlp_and_loading_block_metadata = dlp_and_loading_block_metadata
        self.training_arguments = TrainingArgs(
            training_kwargs, only_required=False
        )
        self.model_arguments = model_kwargs

    def create_round_reply(
        self,
        message: str = "",
        success: bool = False,
        params_url: Optional[str] = None,
        aux_vars_url: Optional[str] = None,
        timing: Optional[Dict[str, float]] = None,
    ) -> TrainReply:
        """Set up a TrainReply to be sent to the researcher.

        This method should be called as part of the `run_model_training`
        method. The message content changes based on the success status.
        Note: if `success=False`, `message` will be logged at error level.

        Args:
            message: Message regarding the process.
            success: Whether training/validation was successful.
            params_url: URL where parameters are uploaded (in case of success).
            aux_vars_url: URL where auxiliary variables are uploaded, if any.
            timing: Timing statistics (in case of training success).

        Returns:
            reply: A TrainReply instance informing the researcher about
                success, with contents documenting how things went.
        """
        # If round is not successful, log the message at error level.
        if not success:
            logger.error(message)
        # Create and return the TrainReply message.
        content = {
            "node_id": environ["NODE_ID"],
            "job_id": self.job_id,
            "researcher_id": self.researcher_id,
            "command": "train",
            "success": success,
            "dataset_id": self.dataset["dataset_id"] if success else "",
            "params_url": params_url or "",
            "aux_vars_url": aux_vars_url,
            "msg": message,
            "timing": timing or {},
        }
        return NodeMessages.reply_create(content)  # type: ignore

    def run(self) -> TrainReply:
        """Run the training (and validation) round.

        This method orchestrates the entire behavior of the `Round`:
        * Download and set up the TrainingPlan with its parameters.
        * Set up the local datasets this TrainingPlan should operate on.
        * Run the training and/or validation steps (before and/or after
          training), depending on the Round's parametrization.
        * Return a TrainReply that documents the success or failure
          of all the former operations.

        Returns:
            reply: A TrainReply instance to send to the researcher,
                documenting whether training was overall successful
                or not, and providing more details based on that.
        """
        # Download and set up the training plan.
        try:
            training_plan = self.setup_training_plan()
        except FedbiomedError as exc:
            return self.create_round_reply(success=False, message=str(exc))
        # Prepare and assign the training and testing data loaders.
        try:
            self.setup_training_plan_data_loaders(training_plan)
        except FedbiomedRoundError as exc:
            return self.create_round_reply(success=False, message=str(exc))
        except Exception as exc:
            msg = f"Uncaught error while setting up data loaders: {exc}"
            return self.create_round_reply(success=False, message=msg)
        # Optionally run a validation round before training.
        testing_arguments = self.training_arguments.testing_arguments()
        if testing_arguments.get("test_on_global_updates"):
            self._run_testing_routine(training_plan, before_train=True)
        # If the training routine is not to be run, send a reply and exit.
        if not self.training:
            return self.create_round_reply(success=True)
        # Try running the training routine.
        try:
            rtime, ptime = time.perf_counter(), time.process_time()
            training_plan.training_routine(
                history_monitor=self.history_monitor, node_args=self.node_args
            )
            timing = {
                "rtime_training": time.perf_counter() - rtime,
                "ptime_training": time.process_time() - ptime,
            }
        # Send a reply and exit on any error.
        except Exception as exc:
            message = f"Failed to train the model: {exc}"
            return self.create_round_reply(success=False, message=message)
        # Optionally run the post-training validation routine.
        if testing_arguments.get("test_on_local_updates"):
            self._run_testing_routine(training_plan, before_train=False)
        # Try uploading model parameters and aux vars to the remote repository.
        try:
            params_url = self._upload_model_params(training_plan)
        except Exception as exc:
            message = f"Failed to upload updated weights: {exc}"
            return self.create_round_reply(success=False, message=message)
        try:
            aux_vars_url = self._upload_optim_aux_vars(training_plan)
        except Exception as exc:
            message = f"Failed to upload optimizer auxiliary variables: {exc}"
            return self.create_round_reply(success=False, message=message)
        # If training and uploading went fine, send a positive reply.
        return self.create_round_reply(
            success=True,
            timing=timing,
            params_url=params_url,
            aux_vars_url=aux_vars_url,
        )

    def setup_training_plan(self) -> TrainingPlan:
        """Try setting up the training plan based on repository files.

        The process implemented here is the following:
        * Download the training plan and parameters files.
        * Optionally check the training plan's approval status.
        * Instantiate the training plan and run its post-init.
        * Reload the training plan's parameters from the file.

        Returns:
            training_plan: The restored TrainingPlan to use.

        Raises:
            FedbiomedRoundError: in case of failure at any step.
        """
        # Try downloading the plan and params files, and check plan approval.
        # Note: a FedbiomedError will be raised in case of failure.
        tplan_path = self._download_file(
            self.training_plan_url, "training_plan"
        )
        self._check_training_plan_approval(tplan_path)
        param_path = self._download_file(self.params_url, "params")
        # Try instantiating the training plan, re-initializing it
        # with model and training args, and reloading its weights.
        try:
            training_plan = TrainingPlan.load_from_json(tplan_path)
            training_plan.post_init(
                model_args=self.model_arguments,
                training_args=self.training_arguments,
            )
            training_plan.load_weights(param_path, assign=True)
        except FedbiomedTrainingPlanError as exc:
            msg = f"Failed to initialize the training plan: {exc}"
            raise FedbiomedRoundError(msg) from exc
        # Optionally try downloading and processing optimizer aux vars.
        if self.aux_vars_url:
            aux_vars_path = self._download_file(self.aux_vars_url, "aux_vars")
            try:
                # NOTE: aux_vars contain dataset-wise auxiliary variables
                #       for the node, hence Round must gather a sub-dict
                aux_vars = declearn.utils.json_load(aux_vars_path)
                uid = f"{environ['NODE_ID']}/{self.dataset['dataset_id']}"
                aux_vars = aux_vars.get(uid, {})
                training_plan.optim.process_aux_var(aux_vars)
            except Exception as exc:
                msg = f"Failed to load an process optimizer aux vars: {exc}"
                raise FedbiomedRoundError(msg) from exc
        # Return the training plan if all went well.
        return training_plan

    def _download_file(
        self,
        url: str,
        basename: str,
    ) -> str:
        """Download a file from a given URL into a file with given basename."""
        # Try downloading the file.
        try:
            filename = f"{basename}_{uuid.uuid4().hex}.json"
            status, path = self.repository.download_file(url, filename)
            if status != 200:
                raise RuntimeError(f"status code {status}")
        # In case of failure, send an error reply and return it.
        except Exception as exc:
            message = f"Failed to download {basename} file from {url}: {exc}"
            raise FedbiomedRoundError(message) from exc
        # In case of success, return the downloaded file's path.
        return path

    def _check_training_plan_approval(self, path: str) -> None:
        """Optionally check the training plan's approval status.

        Whether the status is actually checked or not is controlled by
        the node's environment variables, set based on a config file.

        Args:
            path: Path to the downloaded training plan file.

        Raises:
            FedbiomedError: If the training plan is not approved by the node.
        """
        # Optionally check training plan's approval status.
        if environ["TRAINING_PLAN_APPROVAL"]:
            manager = TrainingPlanSecurityManager()
            approved, info = manager.check_training_plan_status(
                path, TrainingPlanApprovalStatus.APPROVED
            )
            message = f" approved by the node {environ['NODE_ID']}"
            # When not approved, send an error reply and return it.
            if not approved:
                message = f"Requested training plan is not {message}"
                raise FedbiomedError(message)
            # Otherwise, log about the approval.
            logger.info(f"Training plan {info['name']} has been {message}")

    def setup_training_plan_data_loaders(
        self,
        training_plan: TrainingPlan,
    ) -> None:
        """Set up and assign data loaders to a given training plan.

        This method handles the following operations:
        * Build a generic DataManager using `training_plan.training_data(...)`.
        * Build a concrete manager based on `training_plan.data_loader_type()`.
        * Pass along configuration parameters to the DataManager's dataset.
        * Optionally set up a DataLoadingPlan on top of the dataset.
        * Split the dataset into a pair of data loaders for train and test.
        * Assign the latter two data loaders to the training plan.

        Args:
            training_plan: TrainingPlan, the data loaders of which to set
                up and assign.

        Raises:
            FedbiomedRoundError: In case any step of the operation fails.
        """
        # Gather testing parameters.
        testing_arguments = self.training_arguments.testing_arguments()
        test_ratio = testing_arguments.get("test_ratio", 0.0)
        test_global = testing_arguments.get("test_on_global_updates")
        test_local = testing_arguments.get("test_on_local_updates")
        # Inform user about mismatch arguments settings.
        if (test_ratio == 0) and (test_local or test_global):
            logger.warning(
                "Validation routinues will fail due to `test_ratio` being set "
                "to 0. Please adjust its value if you wish validation to run."
            )
        elif (test_ratio != 0) and not (test_local or test_global):
            logger.warning(
                "Validation is not set to be run before nor after training. "
                "`test_ratio` will set apart samples that will be unused."
            )
        # Instantiate and load a DataManager from the training plan.
        data_manager = self._build_data_manager(training_plan)
        # Finalize the configuration of its wrapped dataset object.
        try:
            self._configure_dataset(data_manager)
        except FedbiomedRoundError as exc:
            raise exc
        except FedbiomedError as exc:
            raise FedbiomedRoundError(
                f"{ErrorNumbers.FB314.value}: an error was raised while "
                f"configuring the loaded DataManager's dataset: {exc}"
            ) from exc
        # Split the dataset into a pair of data loaders and assign them.
        try:
            loaders = data_manager.split(test_ratio=test_ratio)
            training_plan.set_data_loaders(*loaders)
        except FedbiomedError as exc:
            raise FedbiomedRoundError(
                f"{ErrorNumbers.FB314.value}: an error was raised while "
                f"setting up and assigning train/test data loaders: {exc}"
            ) from exc

    def _build_data_manager(
        self,
        training_plan: TrainingPlan,
    ) -> TypeDataManager:
        """Set up and load a concrete DataManager based on a TrainingPlan.

        Args:
            training_plan: TrainingPlan, the `training_data` method of
                which to call, and the `data_loader_type()` from which
                to use when calling `DataManager.build`.

        Returns:
            data_manager: Instance created by calling the `build` method
                of the generic `DataManager` returned from the training
                plan's `training_data` method. The latter is called with
                parameters taken from the `dataset` and `training_arguments`
                wrapped by this `Round`.

        Raises:
            FedbiomedRoundError: If the call to `training_data` fails,
                if the returned object is not a `DataManager` or if it
                fails to be loaded.
        """
        # Inspect the arguments of the method `training_data`, because it is
        # defined by the researcher and might include additional arguments.
        signature = inspect.signature(training_plan.training_data)
        arguments = set(signature.parameters.keys())
        # Map training arguments to the former.
        # Separate those passed to `training_data` from those passed later on.
        loader_args = self.training_arguments.loader_arguments()
        arguments = set.intersection(arguments, loader_args)
        t_data_args = {key: loader_args.pop(key) for key in arguments}
        # Try running the method.
        try:
            data_manager = training_plan.training_data(
                dataset_path=self.dataset["path"], **t_data_args
            )
        except Exception as exc:
            raise FedbiomedRoundError(
                f"{ErrorNumbers.FB314.value}: `training_plan.training_data` "
                f"call failed, raising error: {exc}"
            ) from exc
        # Type-check the returned object.
        if not isinstance(data_manager, DataManager):
            raise FedbiomedRoundError(
                f"{ErrorNumbers.FB314.value}: The method `training_data` "
                "should a `fedbiomed.common.data.DataManager` instance, "
                f"not {type(data_manager)}"
            )
        # Build the type-specific data manager from the generic factory.
        try:
            data_manager.set_loader_arguments(loader_args, update=True)
            return data_manager.build(training_plan.data_loader_type())
        except FedbiomedError as exc:
            raise FedbiomedRoundError(
                f"{ErrorNumbers.FB314.value}: Error while building the "
                f"concrete data manager: {exc}"
            ) from exc

    def _configure_dataset(
        self,
        data_manager: TypeDataManager,
    ) -> None:
        """Configure the dataset wrapped by a DataManager class.

        Args:
            data_manager: Data manager freshly created through the DataManager
                generic factory.

        Raises:
            FedbiomedDatasetError: if a method from `data_manager` or from
                `data_manager.dataset` fails and raises one.
            FedbiomedRoundError: if a DataLoadingPlan is configured to be
                set on the `data_manager.dataset` but cannot be.
        """
        if not isinstance(data_manager, TorchDataManager):
            if (
                self.dataset.get("dataset_parameters")
                or self._dlp_and_loading_block_metadata is not None
            ):
                logger.warning(
                    "'dataset_parameters' and 'DataLoadingPlan' parameters "
                    "wrapped by `Round` cannot be applied to the dataset: "
                    "it is not handled through a TorchDataManager."
                )
        # Optionally pass on some dataset parameters.
        if hasattr(data_manager.dataset, "set_dataset_parameters"):
            dataset_parameters = self.dataset.get("dataset_parameters", {})
            data_manager.dataset.set_dataset_parameters(dataset_parameters)
        # Optionally set up a data loading plan.
        if self._dlp_and_loading_block_metadata is not None:
            dlp_cfg, blocks_cfg = self._dlp_and_loading_block_metadata
            if hasattr(data_manager.dataset, "set_dlp"):
                dlp = DataLoadingPlan()
                dlp = dlp.deserialize(dlp_cfg, blocks_cfg)
                data_manager.dataset.set_dlp(dlp)
            else:
                raise FedbiomedRoundError(
                    f"{ErrorNumbers.FB314.value}: Round is configured to set "
                    f"DataLoadingPlan {dlp_cfg['name']} on dataset of type "
                    f"{data_manager.dataset.__class__.__name__} which does "
                    "not support it (no `set_dlp` method)."
                )

    def _run_testing_routine(
        self,
        training_plan: TrainingPlan,
        before_train: bool,
    ) -> None:
        """Run a training plan's testing routine, catching any exception.

        Args:
            training_plan: TrainingPlan, the testing routine of which to run.
            before_train: Whether the plan was trained locally before this
                function was called (this has merely aesthetical effects).
        """
        params_type = "global" if before_train else "local"
        testing_arguments = self.training_arguments.testing_arguments()
        try:
            training_plan.testing_routine(
                metric=testing_arguments.get("test_metric", None),
                metric_args=testing_arguments.get("test_metric_args", {}),
                history_monitor=self.history_monitor,
                before_train=before_train,
            )
        except FedbiomedError as exc:
            logger.error(
                f"{ErrorNumbers.FB314}: During the testing phase on "
                f"{params_type} parameter updates: {exc}"
            )
        except Exception as exc:
            logger.error(
                f"Undetermined error during the testing phase on "
                f"{params_type} parameter updates: {exc}"
            )

    def _upload_model_params(
        self,
        training_plan: TrainingPlan,
    ) -> str:
        """Upload model parameters from a training plan.

        Args:
            training_plan: TrainingPlan, the parameters from which to upload.

        Returns:
            url: URL at which the model parameters file was uploaded.
        """
        filename = os.path.join(
            environ["TMP_DIR"], f"node_params_{uuid.uuid4()}.json"
        )
        training_plan.save_weights(filename)
        repo_resp = self.repository.upload_file(filename)
        logger.info("Updated weights successfully uploaded.")
        return repo_resp["file"]  # type: ignore

    def _upload_optim_aux_vars(
        self,
        training_plan: TrainingPlan,
    ) -> Optional[str]:
        """Collect and upload optimizer auxiliary variables, if any.

        Args:
            training_plan: TrainingPlan, the aux. vars from which to upload.

        Returns:
            url: URL at which the auxiliary variables file was uploaded.
                May be None if there were no auxiliary variables to upload.
        """
        # Collect the auxiliary variables dict, which may be empty.
        aux_vars = training_plan.optim.collect_aux_var()
        if not aux_vars:
            return None
        # Export the dict to a JSON file and upload it.
        filename = os.path.join(
            environ["TMP_DIR"], f"aux_vars_{uuid.uuid4()}.json"
        )
        declearn.utils.json_dump(aux_vars, filename)
        repo_resp = self.repository.upload_file(filename)
        logger.info("Optimizer aux. variables successfully uploaded.")
        return repo_resp["file"]  # type: ignore
