# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Tuple

import torch

from fedbiomed.common.constants import HarmonizationStep
from fedbiomed.researcher.datasets import FederatedDataset
from fedbiomed.researcher.federated_workflows._federated_workflow import (
    FederatedAnalytics,
)

from ._fedcombat_model import _FedCombatTrainModel


class _FedCombatParameters:
    def __init__(
        self,
        experiment_id: str,
        researcher_id: str,
        fds: FederatedDataset,
        reqs,
        experimentation_folder: str,
        nodes: List[str],
    ):
        """
        Class computing the Fed-ComBat steps on the server side
        """
        self._experiment_id = experiment_id
        self._researcher_id = researcher_id
        self._fds = fds
        self._reqs = reqs
        self._experimentation_folder = experimentation_folder
        self._nodes = nodes

        self.step_functions = {
            HarmonizationStep.STANDARDIZE: self._compute_global_mean_std,
            HarmonizationStep.TRAIN_RESID_VAR: self._train_harmonization_model,
            HarmonizationStep.RESID_PARAMS: self._compute_pooled_variance,
            HarmonizationStep.FC_PARAMS: self._compute_residual_parameters,
        }

    def __call__(
        self,
        harmonization_step: HarmonizationStep,
        list_dict_params: List,
        dict_params_local: Optional[dict],
    ):
        """
        Generic call of the class to automatically compute the right function
        for the specified harmonization_step

        Args:
            harmonization_step: Harmonization step Enum allowing to select the right function
            list_dict_params: list of dictionary returned by the nodes
            dict_params_local: dictionary of parameters computed locally. Take precedence
                over the parameters returned by the nodes.

        Returns;
            Dict: parameters resulting from the harmonization_step computation from the researcher
        """
        if dict_params_local is not None:
            kwargs = dict_params_local
        else:
            kwargs = self._stack_dict_params(list_dict_params=list_dict_params)
        return self.step_functions[harmonization_step](**kwargs)

        # Note: at each step, the researcher could check the type and shape of the received parameters.
        # It is not implemented for now to keep implementation simple but could be considered for:
        # - security: avoid malicious parameters. Risk seems limited as they are used for simple math operations,
        #   not function names, etc.
        # - robustness: avoid errors due to wrong parameters. This will be handled by enclosing try/except

    def _compute_global_mean_std(self, **kwargs) -> Tuple[dict, dict]:
        """
        Computes the global means and stds from the nodes' ones

        Args:
            kwargs: Dict containing the list of means and stds of the
                    covariates and phenotypes sent by the nodes.
                    Keys: ["mean_covariates", "mean_phenotypes", "std_covariates", "std_phenotypes"]

        Returns:
            Tuple of Dict containing the global means and stds of the covariates and phenotypes;
                  Keys: ["global_mean_covariates", "global_mean_phenotypes",
                         "global_std_covariates", "global_std_phenotypes"]
            and empty Dict (normally for per node values)
        """
        covariates = kwargs.get("covariates", [])
        phenotypes = kwargs.get("phenotypes", [])

        # Compute mean and std only with active datasets of the federation
        # Needed as FA don't implement node filtering yet
        fds_filtered = FederatedDataset(
            {
                node_id: node_meta
                for node_id, node_meta in self._fds.data().items()
                if node_id in self._nodes
            }
        )

        # Analytics instance specific to this preprocessing as nodes may change from
        # the rest of the experiment & from one call to the other
        analytics = FederatedAnalytics(
            fds=fds_filtered,
            experiment_id=self._experiment_id,
            researcher_id=self._researcher_id,
            reqs=self._reqs,
            experimentation_folder=self._experimentation_folder,
        )

        # FA glitch: currently need to request `variance` to retrieve `std`
        result = analytics.fetch_stats(["mean", "variance"])

        global_mean_covariates = [
            v for k, v in result.global_stats("mean").items() if k in covariates
        ]
        global_mean_phenotypes = [
            v for k, v in result.global_stats("mean").items() if k in phenotypes
        ]
        global_std_covariates = [
            v for k, v in result.global_stats("std").items() if k in covariates
        ]
        global_std_phenotypes = [
            v for k, v in result.global_stats("std").items() if k in phenotypes
        ]

        # Clean up
        del analytics

        return {
            "global_mean_covariates": torch.tensor(
                global_mean_covariates, dtype=torch.float32
            ),
            "global_mean_phenotypes": torch.tensor(
                global_mean_phenotypes, dtype=torch.float32
            ),
            "global_std_covariates": torch.tensor(
                global_std_covariates, dtype=torch.float32
            ),
            "global_std_phenotypes": torch.tensor(
                global_std_phenotypes, dtype=torch.float32
            ),
        }, {}

    def _train_harmonization_model(self, **kwargs) -> Tuple[dict, dict]:
        """Train the harmonization model using the provided arguments.

        Args:
            kwargs: Dict containing the arguments for training the harmonization model.
                    Keys: ["covariates", "phenotypes", "training_args", "model_args", "rounds"]

        Returns:
            Tuple of Dict containing the trained biological model parameters, global bias model parameters, and local bias model parameters by node.
                  Keys: ["biological_model", "global_bias_model"]
                  and Dict containing the local bias model parameters by node.
        """
        covariates = kwargs.get("covariates")
        phenotypes = kwargs.get("phenotypes")
        training_args = kwargs.get("training_args")
        model_args = kwargs.get("model_args")
        rounds = kwargs.get("rounds")

        fc_model_training = _FedCombatTrainModel(
            self._fds,
            self._nodes,
            self._experimentation_folder,
            covariates,
            phenotypes,
            training_args,
            model_args,
            rounds,
        )
        biological_model, global_bias_model, local_bias_models = (
            fc_model_training.execute()
        )

        return {
            "biological_model": biological_model,
            "global_bias_model": global_bias_model,
        }, local_bias_models

    def _compute_pooled_variance(self, **kwargs) -> Tuple[dict, dict]:
        """
        Computes the global residual pooled variance from the nodes' ones

        Args:
            kwargs: Dict containing the list of residual variances and the number of samples sent by the nodes.
                    Keys: ["residual_variance", "n_samples"]

        Returns:
            Tuple of Dict: Dict containing the pooled residual variance (weighted residual variance)
                  Keys: ["sigma_hat_g"]
                and empty Dict (normally for per node values)
        """
        stacked_residual_variances = kwargs["residual_variance"]
        stacked_n_samples = kwargs["n_samples"]

        sigma_hat_g = self._weighted_mean(
            stacked_residual_variances, stacked_n_samples, ddof=1
        )  # pooled variance
        sigma_hat_g[sigma_hat_g == 0] = 1e-8
        return {"sigma_hat_g": sigma_hat_g}, {}

    def _compute_residual_parameters(self, **kwargs) -> Tuple[dict, dict]:
        """
        Computes the global bayesian priors from the nodes' local residual statistics

        Args:
            kwargs: Dict containing the list of the means (gamma_hat_ig) and variances (delta_hat_ig) of the residuals sent by the nodes.
                    Keys: ["gamma_hat_ig", "delta_hat_ig"]

        Returns:
            Dict: Dict containing the computed bayesian priors
                  Keys: ["gamma_bar", "tau_2", "lambda_bar_i", "theta_bar_i"]
                  and empty Dict (normally for per node values)
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
        }, {}

    def _weighted_mean(
        self, values: torch.Tensor, weights: torch.Tensor, ddof: int = 0
    ) -> torch.Tensor:
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

    def _stack_dict_params(self, list_dict_params: list[dict]) -> dict:
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

    ###### DUMMY DATA ######
    shape_phenotypes = torch.rand((100, 2))
    ########################

    #####################################
    ########## DUMMY FUNCTIONS ##########
    #####################################

    def read_biological_model_values(self):
        return self.shape_phenotypes

    def read_bias_model_values(self):
        return torch.zeros_like(self.shape_phenotypes)
