# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from typing import Dict

import torch

from fedbiomed.common.constants import HarmonizationStep


class _FedCombatJobs:
    """
    Class computing the Fed-ComBat steps on the node side
    """

    # TODO: use the right arguments to use non-dummy data (DatasetManager, NodeState, ...)
    def __init__(
        self,
        # root_dir: str,
        # dataset_manager: DatasetManager,
    ):
        # TODO: remove
        ###### DUMMY DATA ######
        torch.manual_seed(42)
        self.covariates = torch.rand((100, 3))
        self.phenotypes = torch.rand((100, 2))
        ########################

        self.n_samples = torch.tensor(self.covariates.shape[0])

        self.step_functions = {
            HarmonizationStep.STEP1_MEAN_STD: self._compute_mean_std,
            HarmonizationStep.STEP2_STANDARDIZE: self._standardize_data,
            # There is no step 3 as it's a model training handled by a training plan
            # HarmonizationStep.STEP3_TRAIN: lambda x: {},
            HarmonizationStep.STEP4_RESID_VAR: self._compute_residual_variance,
            HarmonizationStep.STEP5_RESID_PARAMS: self._compute_standardized_residuals_params,
            HarmonizationStep.STEP6_FC_PARAMS: self._compute_fedcombat_params,
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

    # Note: at each step, the node could check the type and shape of the received parameters.
    # It is not implemented for now to keep implementation simple but could be considered for:
    # - security: avoid malicious parameters. Risk seems limited as they are used for simple math operations,
    #   not function names, etc.
    # - robustness: avoid errors due to wrong parameters. This will be handled by enclosing try/except

    def _compute_mean_std(self, params) -> Dict:
        """
        Computes mean and standard deviation of the covariates and phenotypes

        Args:
            params: empty Dict

        Returns:
            Dict: Dict containing means and stds of the local covariates and phenotypes.
                  Keys: ["mean_covariates", "mean_phenotypes", "std_covariates", "std_phenotypes", "n_samples"]
        """
        means_stds = {
            "mean_covariates": self.covariates.mean(0),
            "mean_phenotypes": self.phenotypes.mean(0),
            "std_covariates": self.covariates.std(0),
            "std_phenotypes": self.phenotypes.std(0),
            "n_samples": self.n_samples,
        }
        return means_stds

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
        global_mean_covariates = params["global_mean_covariates"]
        global_mean_phenotypes = params["global_mean_phenotypes"]
        global_std_covariates = params["global_std_covariates"]
        global_std_phenotypes = params["global_std_phenotypes"]

        self.standardized_covariates = (
            self.covariates - global_mean_covariates
        ) / global_std_covariates
        self.standardized_phenotypes = (
            self.phenotypes - global_mean_phenotypes
        ) / global_std_phenotypes

        # Save the standardized data
        self.save_data_locally(self.standardized_covariates)
        self.save_data_locally(self.standardized_phenotypes)

        return {}

    def _compute_residual_variance(self, params) -> dict:
        """
        Computes the variance of the residuals from the biological model

        Args:
            params: Empty Dict

        Returns:
            Dict: Dict containing the local residual variance and the number of samples
                  Keys: ["residual_variance", "n_samples"]
        """
        ########## TODO Replace by proper functions to read the data from the node ##########
        biological_model = params["biological_model"]
        global_bias = params["global_bias_model"]
        local_bias = self.read_bias_model(1234)

        standardized_covariates = self.read_standardized_covariates(123)
        standardized_phenotypes = self.read_standardized_phenotypes(123)
        #####################################################################################
        bias_param = torch.ones((len(standardized_covariates), 1))
        preds = biological_model - global_bias - local_bias(bias_param)
        residuals = standardized_phenotypes - preds
        residual_variance = residuals.var(0)
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
        ########## TODO: Replace by proper functions to read the data from the node ##########
        biological_model = params["biological_model"]
        global_bias = params["global_bias_model"]
        # standardized_covariates = self.read_standardized_covariates(123)
        standardized_phenotypes = self.read_standardized_phenotypes(123)
        ################################################################################
        sigma_hat_g = params["sigma_hat_g"]

        # TODO: missing "- local_bias" term or bias_param not needed ?
        # bias_param = torch.ones((len(standardized_covariates), 1))
        preds = biological_model - global_bias
        residuals = standardized_phenotypes - preds
        z = residuals / sigma_hat_g

        # Save standardized residuals
        self.save_data_locally(z)
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
        ########## TODO: Replace by proper functions to read the data from the node ##########
        biological_model = params["biological_model"]
        global_bias = params["global_bias_model"]

        # standardized_covariates = self.read_standardized_covariates(123)
        z = self.read_standardized_residuals(1234)
        ################################################################################

        gamma_bar = params["gamma_bar"]
        tau_2 = params["tau_2"]
        lambda_bar_i = params["lambda_bar_i"]
        theta_bar_i = params["theta_bar_i"]
        sigma_hat_g = params["sigma_hat_g"]

        gamma_hat_ig = z.mean(0)
        delta_hat_ig = z.var(0)

        # TODO: missing "- local_bias" term or bias_param not needed ?
        # bias_param = torch.ones((len(standardized_covariates), 1))
        pred = biological_model - global_bias

        # Initial value
        delta2_star_ig = delta_hat_ig

        for _ in range(30):
            gamma_star_ig = (
                self.n_samples * tau_2 * gamma_hat_ig + delta2_star_ig * gamma_bar
            )
            gamma_star_ig /= self.n_samples * tau_2 + delta2_star_ig

            delta2_star_ig = theta_bar_i + 0.5 * torch.sum(
                (z - gamma_star_ig) ** 2, axis=0
            )
            delta2_star_ig /= 0.5 * self.n_samples + lambda_bar_i - 1

        harmonized_data = (
            sigma_hat_g / delta2_star_ig.sqrt() * (z - gamma_star_ig)
        ) + pred
        harmonized_dataset_id = self.save_dataset(harmonized_data)
        return {"harmonized_dataset_id": harmonized_dataset_id}

    #####################################
    ########## DUMMY FUNCTIONS ##########
    #####################################

    def read_standardized_covariates(self, dataset_id):
        return self.covariates

    def read_standardized_phenotypes(self, dataset_id):
        return self.phenotypes

    def read_standardized_residuals(self, dataset_id):
        return self.phenotypes

    def read_biological_model(self, model_id):
        return lambda x: self.phenotypes

    def read_bias_model(self, model_id):
        return lambda x: torch.zeros_like(self.phenotypes)

    def save_dataset(self, dataset):
        new_dataset_id = 12345
        return new_dataset_id

    def save_data_locally(self, data):
        return
