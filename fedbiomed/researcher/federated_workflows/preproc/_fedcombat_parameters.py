from typing import List

import torch

from fedbiomed.common.constants import HarmonizationStep


class _FedComBatParameters:
    def __init__(self):
        """
        Class computing thee Fed-ComBat steps on the server side
        """
        torch.manual_seed(42)
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

        :param harmonization_step: Description
        :type harmonization_step: HarmonizationStep
        :param list_dict_params: Description
        :type list_dict_params: List
        """
        stacked_kwargs = self._stack_dict_params(list_dict_params=list_dict_params)
        return self.step_functions[harmonization_step](**stacked_kwargs)

    def _compute_global_mean_std(self, **kwargs):
        """
        Computes the global means and stds from the nodes' ones

        :param self: Description
        :param kwargs: Description
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

        :param kwargs: Description
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

        :param kwargs: Description
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

        :param values: Description
        :type values: torch.Tensor
        :param weights: Description
        :type weights: torch.Tensor
        :param ddof: Description
        :type ddof: int
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

        :param list_dict_params: Description
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
