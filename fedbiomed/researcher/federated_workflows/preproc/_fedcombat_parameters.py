# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from typing import List

import torch

from fedbiomed.common.constants import HarmonizationStep


class _FedComBatParameters:
    def __init__(self):
        """
        Class computing the Fed-ComBat steps on the server side
        """
        self.step_functions = {
            HarmonizationStep.STEP1: lambda: {},
            HarmonizationStep.STEP2: self._compute_global_mean_std,
            HarmonizationStep.STEP3: lambda: {},
            HarmonizationStep.STEP4: lambda: {},
            HarmonizationStep.STEP5: self._compute_pooled_variance,
            HarmonizationStep.STEP6: self._compute_residual_parameters,
        }

    def __call__(self, harmonization_step: HarmonizationStep, list_dict_params: List):
        """
        Generic call of the class to automatically compute the right function
        for the specified harmonization_step

        Args:
            harmonization_step: Harmonization step Enum allowing to select the right function
            list_dict_params: list of dictionary returned by the nodes

        Returns;
            Dict: parameters resulting from the harmonization_step computation from the researcher
        """
        stacked_kwargs = self._stack_dict_params(list_dict_params=list_dict_params)
        return self.step_functions[harmonization_step](**stacked_kwargs)

    # Note: at each step, the researcher could check the type and shape of the received parameters.
    # It is not implemented for now to keep implementation simple but could be considered for:
    # - security: avoid malicious parameters. Risk seems limited as they are used for simple math operations,
    #   not function names, etc.
    # - robustness: avoid errors due to wrong parameters. This will be handled by enclosing try/except

    def _compute_global_mean_std(self, **kwargs):
        """
        Computes the global means and stds from the nodes' ones

        Args:
            kwargs: Dict containing the list of means and stds of the
                    covariates and phenotypes sent by the nodes.
                    Keys: ["mean_covariates", "mean_phenotypes", "std_covariates", "std_phenotypes"]

        Returns:
            Dict: Dict containing the global means and stds of the covariates and phenotypes;
                  Keys: ["global_mean_covariates", "global_mean_phenotypes",
                         "global_std_covariates", "global_std_phenotypes"]
        """
        stacked_mean_covariates = kwargs["mean_covariates"]
        stacked_mean_phenotypes = kwargs["mean_phenotypes"]

        stacked_std_covariates = kwargs["std_covariates"]
        stacked_std_phenotypes = kwargs["std_phenotypes"]

        stacked_n_samples = kwargs["n_samples"]

        global_mean_covariates = self._weighted_mean(
            stacked_mean_covariates, stacked_n_samples, ddof=0
        )
        global_mean_phenotypes = self._weighted_mean(
            stacked_mean_phenotypes, stacked_n_samples, ddof=0
        )

        global_std_covariates = self._weighted_mean(
            stacked_std_covariates, stacked_n_samples, ddof=1
        )
        global_std_phenotypes = self._weighted_mean(
            stacked_std_phenotypes, stacked_n_samples, ddof=1
        )

        return {
            "global_mean_covariates": global_mean_covariates,
            "global_mean_phenotypes": global_mean_phenotypes,
            "global_std_covariates": global_std_covariates,
            "global_std_phenotypes": global_std_phenotypes,
        }

    def _compute_pooled_variance(self, **kwargs):
        """
        Computes the global residual pooled variance from the nodes' ones

        Args:
            kwargs: Dict containing the list of residual variances and the number of samples sent by the nodes.
                    Keys: ["residual_variance", "n_samples"]

        Returns:
            Dict: Dict containing the pooled residual variance (weighted residual variance)
                  Keys: ["sigma_hat_g"]
        """
        stacked_residual_variances = kwargs["residual_variance"]
        stacked_n_samples = kwargs["n_samples"]

        sigma_hat_g = self._weighted_mean(
            stacked_residual_variances, stacked_n_samples, ddof=1
        )  # pooled variance
        sigma_hat_g[sigma_hat_g == 0] = 1e-8
        return {"sigma_hat_g": sigma_hat_g}

    def _compute_residual_parameters(self, **kwargs):
        """
        Computes the global bayesian priors from the nodes' local residual statistics

        Args:
            kwargs: Dict containing the list of the means (gamma_hat_ig) and variances (delta_hat_ig) of the residuals sent by the nodes.
                    Keys: ["gamma_hat_ig", "delta_hat_ig"]

        Returns:
            Dict: Dict containing the computed bayesian priors
                  Keys: ["gamma_bar", "tau_2", "lambda_bar_i", "theta_bar_i"]
        """
        stacked_gamma_hat_ig = kwargs["gamma_hat_ig"]
        stacked_delta_hat_ig = kwargs["delta_hat_ig"]

        gamma_bar = stacked_gamma_hat_ig.mean(0)
        tau_2 = stacked_gamma_hat_ig.var(0)

        v_bar = stacked_delta_hat_ig.mean(0)
        s_bar_2 = stacked_delta_hat_ig.var(0)

        s_bar_2[s_bar_2 == 0.0] = 1e-8

        lambda_bar_i = (v_bar + 2 * s_bar_2) / s_bar_2
        theta_bar_i = (v_bar**3 + v_bar * s_bar_2) / s_bar_2

        return {
            "gamma_bar": gamma_bar,
            "tau_2": tau_2,
            "lambda_bar_i": lambda_bar_i,
            "theta_bar_i": theta_bar_i,
        }

    def _weighted_mean(
        self, values: torch.Tensor, weights: torch.Tensor, ddof: int = 0
    ):
        """
        Computes the weighted mean from a set of values and weights.

        Args:
            values: a torch.Tensor of size (N x M) containing the values to average
            weights: a torch.Tensor of size (N) containing the weights
            ddof: an int to subtract the degrees of freedom (useful for unbiased variance)

        Returns:
            torch.Tensor: computed weighted mean of shape (1 x M)
        """
        weighted_sum = torch.vstack(
            [val * (weight - ddof) for val, weight in zip(values, weights, strict=True)]
        )
        weighted_sum = weighted_sum.sum(0)
        weights_sum = weights.sum()
        weighted_mean = weighted_sum / (weights_sum - ddof * len(weights))
        return weighted_mean

    def _stack_dict_params(self, list_dict_params):
        """
        Stacks the values from a list of dictionaries into a single one.

        Args:
            list_dict_params: list of Dicts (output parameters returned by the nodes)

        Returns:
            Dict: containing the stacked values from the list of Dicts
        """
        if len(list_dict_params) == 0:
            return {}

        param_keys = list(list_dict_params[0].keys())
        stacked_params = {
            param_key: torch.vstack(
                [dict_param[param_key] for dict_param in list_dict_params]
            )
            for param_key in param_keys
        }
        return stacked_params
