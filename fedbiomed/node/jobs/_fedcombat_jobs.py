# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from typing import Any, Dict, Tuple

import polars as pl
import torch

from fedbiomed.common.constants import (
    HARMONIZED_PREFIX,
    NODE_DYNAMIC_DATA_FOLDER,
    UPDATED_PREFIX,
    HarmonizationStep,
)
from fedbiomed.common.dataset import TabularDataset
from fedbiomed.common.dataset_controller import TabularController
from fedbiomed.common.dataset_types import DataReturnFormat
from fedbiomed.common.preproc import FedCombatBiasModel, FedCombatBiologicalModel
from fedbiomed.common.serializer import Serializer
from fedbiomed.node.dataset_manager import DatasetManager
from fedbiomed.node.node_state_manager import NodeStateFileName, NodeStateManager


class _FedCombatJobs:
    """
    Class computing the Fed-ComBat steps on the node side

    Current implementation makes the assumption the whole dataset can be loaded in memory.
    """

    def __init__(
        self,
        root_dir: str,
        experiment_id: str,
        preproc_id: str,
        researcher_id: str,
        node_state_manager: NodeStateManager,
        dataset_manager: DatasetManager,
        dataset_entry: dict,
    ):
        """Constructor of the class

        Args:
            root_dir: Root fedbiomed directory where node instance files will be stored.
            experiment_id: id of the experiment to which the preprocessing belongs
            preproc_id: id of the preprocessing to which the preprocessing belongs
            researcher_id: id of the researcher who launched the preprocessing
            node_state_manager: initialized NodeStateManager instance to save and retrieve data during the preprocessing steps
            dataset_manager: DatasetManager instance to retrieve datasets
            dataset_entry: dict containing the dataset entry information from the dataset registry for the dataset to preprocess
        """
        self._root_dir = root_dir
        self._experiment_id = experiment_id
        self._preproc_id = preproc_id
        self._researcher_id = researcher_id
        self._node_state_manager = node_state_manager
        self._dataset_manager = dataset_manager
        self._dataset_entry = dataset_entry

        self.step_functions = {
            HarmonizationStep.STANDARDIZE: self._standardize_data,
            HarmonizationStep.TRAIN_RESID_VAR: self._compute_residual_variance,
            HarmonizationStep.RESID_PARAMS: self._compute_standardized_residuals_params,
            HarmonizationStep.FC_PARAMS: self._compute_fedcombat_params,
        }

    def __call__(
        self,
        harmonization_step: HarmonizationStep,
        dataset_type: str,
        params: Dict,
    ) -> Dict:
        """
        Generic call of the class to automatically compute the right function
        for the specified harmonization_step

        Args:
            harmonization_step: Harmonization step Enum allowing to select the right function
            dataset_type: type of the dataset to harmonize
            params: Dictionary containing the parameters to pass to the called function

        Returns:
            Dict: parameters resulting from the harmonization_step computation from the node

        Raises:
            ValueError: if the dataset type is not supported
        """
        self._harmonization_step = harmonization_step

        if dataset_type != "csv":
            # In this case we don't need to raise a FedbiomedError as it is caught in calling function
            raise ValueError(
                f"FedCombat dataset type {dataset_type} not supported. Expected CSV dataset."
            )

        return self.step_functions[harmonization_step](params)

    # Functions for each step of the Fed-ComBat harmonization

    # Note: at each step, the node could check the type and shape of the received parameters.
    # It is not implemented for now to keep implementation simple but could be considered for:
    # - security: avoid malicious parameters. Risk seems limited as they are used for simple math operations,
    #   not function names, etc.
    # - robustness: avoid errors due to wrong parameters. This will be handled by enclosing try/except

    def _standardize_data(self, params) -> dict:
        """
        Standardizes the data from given means and standard deviations

        Args:
            params: Dict containing the name of the covariate and phenotype variables,
                as well as the global means and stds of the covariates and phenotypes.
                    Keys: [
                    "covariates", "phenotypes",
                    "global_mean_covariates", "global_mean_phenotypes",
                           "global_std_covariates", "global_std_phenotypes"]

        Returns:
            Dict containing the description of the standardized dataset
                Keys: ["standardized_dataset"]
        """
        # Parameters received from the researcher
        covariates_name = params["covariates"]
        phenotypes_name = params["phenotypes"]
        global_mean_covariates = params["global_mean_covariates"]
        global_mean_phenotypes = params["global_mean_phenotypes"]
        global_std_covariates = params["global_std_covariates"]
        global_std_phenotypes = params["global_std_phenotypes"]

        # Read covariates and phenotypes from dataset
        covariates, phenotypes = self._read_initial_dataset_values(
            covariates_name, phenotypes_name
        )
        n_samples = covariates.shape[0]

        # Compute standardized covariates and phenotypes
        standardized_covariates = (
            covariates - global_mean_covariates
        ) / global_std_covariates
        standardized_phenotypes = (
            phenotypes - global_mean_phenotypes
        ) / global_std_phenotypes

        # Save temporary dataset with standardized values
        # for training FedComBat models
        standardized_dataset = self._save_updated_dataset_values(
            phenotypes_name=phenotypes_name,
            phenotypes_data=standardized_phenotypes,
            covariates_name=covariates_name,
            covariates_data=standardized_covariates,
            is_harmonization=False,
        )

        # Save results locally
        state = {
            "covariates_name": covariates_name,
            "phenotypes_name": phenotypes_name,
            "n_samples": n_samples,
            "standardized_covariates": standardized_covariates,
            "standardized_phenotypes": standardized_phenotypes,
            "std_dataset_id": standardized_dataset["dataset_id"],
            "global_mean_phenotypes": global_mean_phenotypes,
            "global_std_phenotypes": global_std_phenotypes,
        }
        self._save_state_values(state)

        # Return results to researcher
        return {"standardized_dataset": standardized_dataset}

    def _compute_residual_variance(self, params) -> dict:
        """
        Computes the variance of the residuals from the biological model

        Args:
            params: Dict containing the parameters from the trained models
                    Keys: ["biological_model", "global_bias_model", "local_bias_model"]

        Returns:
            Dict: Dict containing the local residual variance and the number of samples
                  Keys: ["residual_variance", "n_samples"]
        """
        # Parameters received from the researcher
        biological_model_params = params["biological_model"]
        global_bias_model_params = params["global_bias_model"]
        local_bias_model_params = params["local_bias_model"]

        # Parameters read locally
        state = self._read_state_values()
        global_mean_phenotypes = state["global_mean_phenotypes"]
        global_std_phenotypes = state["global_std_phenotypes"]
        covariates_name = state["covariates_name"]
        phenotypes_name = state["phenotypes_name"]
        n_samples = int(state["n_samples"])
        standardized_covariates = torch.as_tensor(state["standardized_covariates"])
        standardized_phenotypes = torch.as_tensor(state["standardized_phenotypes"])
        bias_param = torch.ones((n_samples, 1))

        n_covariates = len(covariates_name)
        n_phenotypes = len(phenotypes_name)

        # Clean temporary dataset with standardized values used for training FedComBat models
        #
        # Note: we can trust `standardized_dataset_id` as it is generated by the node itself
        # and not provided by the researcher
        standardized_dataset_id = state["std_dataset_id"]
        self._delete_temporary_dataset(standardized_dataset_id)

        # Instantiate models and load parameters
        biological_model = FedCombatBiologicalModel(
            n_covariates=n_covariates, n_phenotypes=n_phenotypes
        )
        biological_model.load_state_dict(biological_model_params)
        global_bias_model = FedCombatBiasModel(n_phenotypes=n_phenotypes)
        global_bias_model.load_state_dict(global_bias_model_params)
        local_bias_model = FedCombatBiasModel(n_phenotypes=n_phenotypes)
        local_bias_model.load_state_dict(local_bias_model_params)

        with torch.no_grad():
            # Infer models on local data
            biological_model_values = biological_model(standardized_covariates)
            global_bias_values = global_bias_model(bias_param)
            local_bias_values = local_bias_model(bias_param)

            # Compute residual variance
            preds = biological_model_values + global_bias_values + local_bias_values
            residuals = standardized_phenotypes - preds
            residual_variance = residuals.var(0)

        # Save results locally
        state = {
            "phenotypes_name": phenotypes_name,
            "n_samples": n_samples,
            "standardized_phenotypes": standardized_phenotypes,
            "biological_model_values": biological_model_values,
            "global_bias_values": global_bias_values,
            "global_mean_phenotypes": global_mean_phenotypes,
            "global_std_phenotypes": global_std_phenotypes,
        }
        self._save_state_values(state)

        # Return results to researcher
        return {
            "residual_variance": residual_variance,
            "n_samples": torch.tensor(n_samples),
        }

    def _compute_standardized_residuals_params(self, params) -> dict:
        """
        Computes the standardized residuals and returns their means and variances

        Args:
            params: Dict containing the global pooled variance of the residuals
                    Keys: ["sigma_hat_g"]

        Returns:
            Dict: Dict containing the mean and the variance of the standardized residuals
                  Keys: ["gamma_hat_ig", "delta_hat_ig"]
        """
        # Parameters received from the researcher
        sigma_hat_g = params["sigma_hat_g"]

        # Parameters read locally
        state = self._read_state_values()
        global_mean_phenotypes = torch.as_tensor(state["global_mean_phenotypes"])
        global_std_phenotypes = torch.as_tensor(state["global_std_phenotypes"])
        phenotypes_name = list(state["phenotypes_name"])
        n_samples = int(state["n_samples"])
        standardized_phenotypes = torch.as_tensor(state["standardized_phenotypes"])
        biological_model_values = torch.as_tensor(state["biological_model_values"])
        global_bias_values = torch.as_tensor(state["global_bias_values"])

        # Compute standardized residuals parameters
        preds = biological_model_values + global_bias_values
        residuals = standardized_phenotypes - preds
        z = residuals / sigma_hat_g

        # Save results locally
        state = {
            "phenotypes_name": phenotypes_name,
            "n_samples": n_samples,
            "sigma_hat_g": sigma_hat_g,
            "standardized_residuals": z,
            "biological_model_values": biological_model_values,
            "global_bias_values": global_bias_values,
            "global_mean_phenotypes": global_mean_phenotypes,
            "global_std_phenotypes": global_std_phenotypes,
        }
        self._save_state_values(state)
        # Return results to researcher
        return {"gamma_hat_ig": z.mean(0), "delta_hat_ig": z.var(0)}

    def _compute_fedcombat_params(self, params) -> dict:
        """
        Estimates the ComBat parameters and harmonizes the dataset

        Args:
            params: Dict containing the bayesian priors to estimate the ComBat parameters, and
                whether the harmonized dataset should be standardized or not.
                    Keys: ["gamma_bar", "tau_2", "lambda_bar_i", "theta_bar_i", "standardize_result"]

        Returns:
            Dict: Dict containing the description of the local harmonized dataset
                  Keys: ["harmonized_dataset"]
        """
        # Parameters received from the researcher
        gamma_bar = params["gamma_bar"]
        tau_2 = params["tau_2"]
        lambda_bar_i = params["lambda_bar_i"]
        theta_bar_i = params["theta_bar_i"]
        standardize_result = params["standardize_result"]

        # Parameters read locally
        state = self._read_state_values()
        global_mean_phenotypes = torch.as_tensor(state["global_mean_phenotypes"])
        global_std_phenotypes = torch.as_tensor(state["global_std_phenotypes"])
        phenotypes_name = list(state["phenotypes_name"])
        n_samples = int(state["n_samples"])
        sigma_hat_g = torch.as_tensor(state["sigma_hat_g"])
        z = torch.as_tensor(state["standardized_residuals"])
        biological_model_values = torch.as_tensor(state["biological_model_values"])
        global_bias_values = torch.as_tensor(state["global_bias_values"])

        gamma_hat_ig = z.mean(0)
        delta_hat_ig = z.var(0)

        pred = biological_model_values + global_bias_values

        # Initial value
        delta2_star_ig = delta_hat_ig

        # Iterate to estimate the ComBat parameters
        # Note: could be improved by checking convergence or making
        # the number of iterations a parameter
        for _ in range(30):
            gamma_star_ig = (
                n_samples * tau_2 * gamma_hat_ig + delta2_star_ig * gamma_bar
            )
            gamma_star_ig /= n_samples * tau_2 + delta2_star_ig

            # `dim` is the native PyTorch name for `axis`
            delta2_star_ig = theta_bar_i + 0.5 * torch.sum(
                (z - gamma_star_ig) ** 2, dim=0
            )
            delta2_star_ig /= 0.5 * n_samples + lambda_bar_i - 1

        residuals = sigma_hat_g / delta2_star_ig.sqrt() * (z - gamma_star_ig)
        harmonized_data = residuals + pred

        # Undo standardization to return harmonized dataset in original scale
        if not standardize_result:
            harmonized_data = (
                harmonized_data * global_std_phenotypes
            ) + global_mean_phenotypes

        # Save results locally in new dataset and return new dataset id to researcher
        harmonized_dataset = self._save_updated_dataset_values(
            phenotypes_name, harmonized_data
        )

        return {"harmonized_dataset": harmonized_dataset}

    # Functions for reading and saving dataset
    def _read_initial_dataset_values(
        self,
        covariates_name: list[str],
        phenotypes_name: list[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reads covariates and phenotypes from the original dataset

        Args:
            covariates_name: the names of the covariates columns
            phenotypes_name: the names of the phenotypes columns

        Returns:
            a tuple containing the covariates and phenotypes tensors
        """
        dataset = TabularDataset(
            input_columns=covariates_name,
            target_columns=phenotypes_name,
        )
        dataset.load(to_format=DataReturnFormat.TORCH, root=self._dataset_entry["path"])

        covariates = torch.stack([dataset[i][0] for i in range(len(dataset))])
        phenotypes = torch.stack([dataset[i][1] for i in range(len(dataset))])

        return covariates, phenotypes

    def _delete_temporary_dataset(self, dataset_id: str):
        """
        Deletes a temporary dataset from the dataset registry and the local storage.

        Args:
            dataset_id: the id of the dataset to delete
        """
        dataset_entry, _ = self._dataset_manager.get_dataset_entry_by_id(dataset_id)
        local_path = Path(dataset_entry["path"])
        if local_path.exists():
            local_path.unlink()

        self._dataset_manager.delete_dataset_by_id(dataset_id)

    def _save_updated_dataset_values(
        self,
        phenotypes_name: list[str],
        phenotypes_data: torch.Tensor,
        covariates_name: list[str] | None = None,
        covariates_data: torch.Tensor | None = None,
        is_harmonization: bool = True,
    ) -> dict:
        """
        Saves the updated dataset locally and returns its description.

        Updated dataset has same structure as the original dataset but replaces the phenotypes with the updated phenotypes,
        and optionally replaces the covariates with the updated covariates if provided.
        It is saved as a new dynamic dataset linked to the original dataset in the dataset registry.

        Args:
            phenotypes_name: the names of the phenotypes columns
            phenotypes_data: the phenotypes data to save
            covariates_name: optional names of covariates columns to overwrite in output dataset
            covariates_data: optional covariates data to save
            is_harmonization: whether the updated dataset is a harmonized dataset or generic updated

        Returns:
            dict: the description of the saved updated dataset
        """
        # Build full new updated dataset by replacing phenotypes columns
        # with updated phenotypes in original dataset
        controller = TabularController(root=self._dataset_entry["path"])
        source_df = pl.concat(
            [controller.get_sample(i) for i in range(len(controller))]
        )

        phenotypes_array = phenotypes_data.detach().cpu().numpy()
        for col_idx, phenotype_col in enumerate(phenotypes_name):
            source_df = source_df.with_columns(
                pl.Series(
                    name=phenotype_col,
                    values=phenotypes_array[:, col_idx],
                )
            )

        # Optionally overwrite covariates columns with provided values.
        covariates_name = covariates_name or []
        covariates_data = (
            covariates_data if covariates_data is not None else torch.empty(0)
        )
        if covariates_name and covariates_data.numel() > 0:
            covariates_array = covariates_data.detach().cpu().numpy()
            for col_idx, covariate_col in enumerate(covariates_name):
                source_df = source_df.with_columns(
                    pl.Series(
                        name=covariate_col,
                        values=covariates_array[:, col_idx],
                    )
                )

        if is_harmonization:
            prefix = HARMONIZED_PREFIX
            name_prefix = "FedComBat"
            description_prefix = "Harmonized with FedComBat"
        else:
            prefix = UPDATED_PREFIX
            name_prefix = "Updated"
            description_prefix = "Updated dataset"

        # Save updated dataset as new CSV file
        # Possible improvement: use a dedicated class that can be reused in other module
        source_path = Path(self._dataset_entry["path"])
        base_name = source_path.stem.split(f"_{prefix}")[0]
        updated_path = os.path.join(
            self._root_dir,
            NODE_DYNAMIC_DATA_FOLDER,
            f"{base_name}_{prefix}{self._preproc_id}.csv",
        )
        source_df.write_csv(updated_path)

        # Save dataset entry in database and return new dataset id
        dynamic_dataset_id = self._dataset_manager.add_dynamic_dataset(
            path=str(updated_path),
            researcher_id=self._researcher_id,
            experiment_id=self._experiment_id,
            processing_id=self._preproc_id,
            parent_dataset_id=self._dataset_entry["dataset_id"],
            name=f"{name_prefix} from {self._dataset_entry['name']}",
            tags=None,
            description=f"{description_prefix} from {self._dataset_entry['description']}",
            dataset_id=None,
            dataset_parameters=self._dataset_entry.get("dataset_parameters", {}),
        )

        # Retrieve the dataset entry and remove sensitive information before returning it
        dynamic_dataset_entry, _ = self._dataset_manager.get_dataset_entry_by_id(
            dynamic_dataset_id
        )
        self._dataset_manager.obfuscate_private_information([dynamic_dataset_entry])
        return dynamic_dataset_entry

    # Functions for reading and saving state values during the Fed-ComBat steps, using the NodeStateManager

    def _save_state_values(self, values: dict[str, torch.Tensor]):
        """
        Saves the state values in the node state manager for later use.

        Args:
            values: dict containing the values to save
        """

        preproc_path = self._node_state_manager.generate_folder_and_create_file_name(
            self._experiment_id,
            self._harmonization_step.value,
            NodeStateFileName.PREPROC,
        )
        Serializer.dump(values, preproc_path)

        state = {
            "preproc_state": {
                "preproc_params_path": preproc_path,
            }
        }
        self._node_state_manager.add(self._experiment_id, state)

    def _read_state_values(self) -> dict[str, Any]:
        """
        Reads the state values from the node state manager.

        Returns:
            dict containing the state values read from the node state manager
        """
        state = self._node_state_manager.get(
            self._experiment_id, self._node_state_manager._previous_state_id
        )

        preproc_path = state["preproc_state"]["preproc_params_path"]
        state_values = Serializer.load(preproc_path)

        return state_values
