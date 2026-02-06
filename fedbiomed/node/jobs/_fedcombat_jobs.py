from typing import Dict

import torch

from fedbiomed.common.constants import HarmonizationStep


class _FedComBat_jobs:
    def __init__(
        self,
        # root_dir: str,
        # dataset_manager: DatasetManager,
    ):
        """
        Class computing the Fed-ComBat steps on the node side
        """
        torch.manual_seed(42)
        self.covariates = torch.rand((100, 3))
        self.phenotypes = torch.rand((100, 2))
        self.standardized_covariates = None
        self.standardized_phenotypes = None
        self.z = None
        self.n_samples = torch.tensor(self.covariates.shape[0])

        self.step_functions = {
            HarmonizationStep.STEP1: self._compute_mean_std,
            HarmonizationStep.STEP2: self._standardize_data,
            HarmonizationStep.STEP4: self._compute_residual_variance,
            HarmonizationStep.STEP5: self._compute_standardized_residuals_params,
            HarmonizationStep.STEP6: self._compute_fedcombat_params,
        }

    def __call__(self, harmonization_step: HarmonizationStep, params: Dict):
        """
        Generic call of the class to automatically compute the right function
        for the specified harmonization_step

        :param harmonization_step: Description
        :type harmonization_step: HarmonizationStep
        :param params: Description
        :type params: Dict
        """
        torch.manual_seed(42)
        self.covariates = torch.rand((100, 3))
        self.phenotypes = torch.rand((100, 2))

        return self.step_functions[harmonization_step](params)

    def _compute_mean_std(self, params):
        """
        Computes mean and standard deviation of the covariates and phenotypes
        """
        means_stds = {
            "mean_covariates": self.covariates.mean(0),
            "mean_phenotypes": self.phenotypes.mean(0),
            "std_covariates": self.covariates.std(0),
            "std_phenotypes": self.phenotypes.std(0),
            "n_samples": self.n_samples,
        }
        return means_stds

    def _standardize_data(self, params):
        """
        Standardizes the data from given means and standard deviations

        Args:
            params:
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
        return {}

    def _compute_residual_variance(self, params):
        """
        Computes the variance of the residuals of the biological model

        :param params: Description
        """
        biological_model = self.read_biological_model(params["biological_model_id"])
        global_bias = self.read_bias_model(params["global_bias_model_id"])
        local_bias = self.read_bias_model(params["local_bias_model_id"])

        standardized_covariates = self.read_standardized_covariates(123)
        standardized_phenotypes = self.read_standardized_phenotypes(123)

        bias_param = torch.ones((len(standardized_covariates), 1))
        preds = (
            biological_model(standardized_covariates)
            - global_bias(bias_param)
            - local_bias(bias_param)
        )
        residuals = standardized_phenotypes - preds
        residual_variance = residuals.var(0)
        return {"residual_variance": residual_variance, "n_samples": self.n_samples}

    def _compute_standardized_residuals_params(self, params):
        """
        Computes the standardized residuals and returns their means and variances

        :param params: Description
        """
        biological_model = self.read_biological_model(params["biological_model_id"])
        global_bias = self.read_bias_model(params["global_bias_model_id"])
        sigma_hat_g = params["sigma_hat_g"]

        standardized_covariates = self.read_standardized_covariates(123)
        standardized_phenotypes = self.read_standardized_phenotypes(123)

        bias_param = torch.ones((len(standardized_covariates), 1))
        preds = biological_model(standardized_covariates) - global_bias(bias_param)
        residuals = standardized_phenotypes - preds
        z = residuals / sigma_hat_g

        return {"gamma_hat_ig": z.mean(0), "delta_hat_ig": z.var(0)}

    def _compute_fedcombat_params(self, params):
        """
        Estimates the ComBat parameters and harmonizes the dataset

        :param params: Description
        """
        gamma_bar = params["gamma_bar"]
        tau_2 = params["tau_2"]
        lambda_bar_i = params["lambda_bar_i"]
        theta_bar_i = params["theta_bar_i"]
        sigma_hat_g = params["sigma_hat_g"]

        biological_model = self.read_biological_model(params["biological_model_id"])
        global_bias = self.read_bias_model(params["global_bias_model_id"])

        standardized_covariates = self.read_standardized_covariates(123)
        bias_param = torch.ones((len(standardized_covariates), 1))
        pred = biological_model(standardized_covariates) - global_bias(bias_param)

        z = self.read_standardized_residuals(1234)

        gamma_hat_ig = z.mean(0)
        delta_hat_ig = z.var(0)

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
