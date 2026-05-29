# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from typing import Dict

import torch

from fedbiomed.common.constants import HarmonizationStep
from fedbiomed.common.preproc import FedCombatBiologicalModel
from fedbiomed.common.serializer import Serializer
from fedbiomed.node.node_state_manager import NodeStateFileName, NodeStateManager


class _FedCombatJobs:
    """
    Class computing the Fed-ComBat steps on the node side

    Current implementation makes the assumption the whole dataset can be loaded in memory.
    """

    # TODO: use the right arguments to use non-dummy data (DatasetManager, NodeState, ...)
    def __init__(
        self,
        experiment_id: str,
        node_state_manager: NodeStateManager,
        # root_dir: str,
        # dataset_manager: DatasetManager,
    ):
        """Constructor of the class

        Args:
            experiment_id: id of the experiment to which the preprocessing belongs
            node_state_manager: initialized NodeStateManager instance to save and retrieve data during the preprocessing steps
        """
        # TODO: remove
        ###### DUMMY DATA ######
        torch.manual_seed(42)
        self.covariates = torch.rand((100, 3))
        self.phenotypes = torch.rand((100, 2))
        ########################
        self.n_samples = torch.tensor(self.covariates.shape[0])

        self._experiment_id = experiment_id
        self._node_state_manager = node_state_manager

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
            params: Dict containing global means and stds of the covariates and phenotypes.
                    Keys: ["global_mean_covariates", "global_mean_phenotypes",
                           "global_std_covariates", "global_std_phenotypes"]

        Returns:
            Dict: empty Dict
        """
        # Parameters received from the researcher
        global_mean_covariates = params["global_mean_covariates"]
        global_mean_phenotypes = params["global_mean_phenotypes"]
        global_std_covariates = params["global_std_covariates"]
        global_std_phenotypes = params["global_std_phenotypes"]

        # TODO: read real covariates and phenotypes here from dataset

        standardized_covariates = (
            self.covariates - global_mean_covariates
        ) / global_std_covariates
        standardized_phenotypes = (
            self.phenotypes - global_mean_phenotypes
        ) / global_std_phenotypes

        # Save results locally
        state = {
            "standardized_covariates": standardized_covariates,
            "standardized_phenotypes": standardized_phenotypes,
        }
        self._save_state_values(state)

        # Return results to researcher
        return {}

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

        ########## TODO Replace by proper functions to retrieve data from the researcher ##########
        # local_bias = self._read_bias_model(1234)
        #####################################################################################

        # Parameters read locally
        state = self._read_state_values()
        standardized_covariates = state["standardized_covariates"]
        standardized_phenotypes = state["standardized_phenotypes"]
        bias_param = torch.ones((len(standardized_covariates), 1))

        import remote_pdb

        remote_pdb.RemotePdb("localhost", 4444).set_trace()

        # Caveat : hardcoded biological and global bias models
        # need to match the ones used for training

        # biological_model = torch.nn.Linear(
        #    self.covariates.shape[1], self.phenotypes.shape[1], bias=False
        # )
        # biological_model.weight = torch.nn.Parameter(biological_model_params)
        biological_model = FedCombatBiologicalModel(
            n_covariates=self.covariates.shape[1], n_phenotypes=self.phenotypes.shape[1]
        )
        biological_model.load_state_dict(biological_model_params)

        global_bias_model = torch.nn.Linear(1, self.phenotypes.shape[1], bias=False)
        global_bias_model.weight = torch.nn.Parameter(global_bias_model_params)
        local_bias_model = torch.nn.Linear(1, self.phenotypes.shape[1], bias=False)
        local_bias_model.weight = torch.nn.Parameter(local_bias_model_params)

        with torch.no_grad():
            # Infer models on local data
            biological_model_values = biological_model(standardized_covariates)
            global_bias_values = global_bias_model(bias_param)
            local_bias_values = local_bias_model(bias_param)

            # Compute residual variance
            # TODO: check sign ?
            preds = biological_model_values - global_bias_values - local_bias_values
            residuals = standardized_phenotypes - preds
            residual_variance = residuals.var(0)

        # Save results locally
        state = {
            "standardized_phenotypes": standardized_phenotypes,
            "biological_model_values": biological_model_values,
            "global_bias_values": global_bias_values,
        }
        self._save_state_values(state)

        # Return results to researcher
        return {"residual_variance": residual_variance, "n_samples": self.n_samples}

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
        standardized_phenotypes = state["standardized_phenotypes"]
        biological_model_values = state["biological_model_values"]
        global_bias_values = state["global_bias_values"]

        # Compute standardized residuals parameters

        # TODO: missing "- local_bias" term or bias_param not needed ?
        # bias_param = torch.ones((len(standardized_covariates), 1))
        preds = biological_model_values - global_bias_values
        residuals = standardized_phenotypes - preds
        z = residuals / sigma_hat_g

        # Save results locally
        state = {
            "sigma_hat_g": sigma_hat_g,
            "standardized_residuals": z,
            "biological_model_values": biological_model_values,
            "global_bias_values": global_bias_values,
        }
        self._save_state_values(state)
        # Return results to researcher
        return {"gamma_hat_ig": z.mean(0), "delta_hat_ig": z.var(0)}

    def _compute_fedcombat_params(self, params) -> dict:
        """
        Estimates the ComBat parameters and harmonizes the dataset

        Args:
            params: Dict containing the bayesian priors to estimate the ComBat parameters
                    Keys: ["gamma_bar", "tau_2", "lambda_bar_i", "theta_bar_i", "sigma_hat_g"]

        Returns:
            Dict: Dict containing the id of the harmonized version of the local data
                  Keys: ["harmonized_dataset_id"]
        """
        # Parameters received from the researcher
        gamma_bar = params["gamma_bar"]
        tau_2 = params["tau_2"]
        lambda_bar_i = params["lambda_bar_i"]
        theta_bar_i = params["theta_bar_i"]

        # Parameters read locally
        state = self._read_state_values()
        sigma_hat_g = state["sigma_hat_g"]
        z = state["standardized_residuals"]
        biological_model_values = state["biological_model_values"]
        global_bias_values = state["global_bias_values"]

        gamma_hat_ig = z.mean(0)
        delta_hat_ig = z.var(0)

        # TODO: missing "- local_bias" term or bias_param not needed ?
        # bias_param = torch.ones((len(standardized_covariates), 1))
        pred = biological_model_values - global_bias_values

        # Initial value
        delta2_star_ig = delta_hat_ig

        for _ in range(30):
            gamma_star_ig = (
                self.n_samples * tau_2 * gamma_hat_ig + delta2_star_ig * gamma_bar
            )
            gamma_star_ig /= self.n_samples * tau_2 + delta2_star_ig

            # `dim` is the native PyTorch name for `axis`
            delta2_star_ig = theta_bar_i + 0.5 * torch.sum(
                (z - gamma_star_ig) ** 2, dim=0
            )
            delta2_star_ig /= 0.5 * self.n_samples + lambda_bar_i - 1

        harmonized_data = (
            sigma_hat_g / delta2_star_ig.sqrt() * (z - gamma_star_ig)
        ) + pred

        # Save results locally in new dataset and return new dataset id to researcher
        harmonized_dataset_id = self._save_dataset(harmonized_data)
        return {"harmonized_dataset_id": harmonized_dataset_id}

    # Functions for reading and saving state values during the Fed-ComBat steps, using the NodeStateManager

    def _save_state_values(self, values: dict[str, torch.Tensor]):
        """
        Saves the state values in the node state manager for later use.

        Args:
            values: dict containing the values to save
        """

        preproc_path = self._node_state_manager.generate_folder_and_create_file_name(
            self._experiment_id,
            0,  # not used
            NodeStateFileName.PREPROC,
        )
        Serializer.dump(values, preproc_path)

        state = {
            "preproc_state": {
                "preproc_params_path": preproc_path,
            }
        }
        self._node_state_manager.add(self._experiment_id, state)

    def _read_state_values(self) -> dict[str, torch.Tensor]:
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

    #####################################
    ########## DUMMY FUNCTIONS ##########
    #####################################

    def _read_bias_model(self, model_id):
        return lambda x: torch.zeros_like(self.phenotypes)

    def _save_dataset(self, dataset):
        new_dataset_id = 12345
        return new_dataset_id
