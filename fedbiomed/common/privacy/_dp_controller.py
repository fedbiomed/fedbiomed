import torch

from typing import Union, Dict, Any
from copy import deepcopy
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from fedbiomed.common.validator import ValidateError
from fedbiomed.common.training_args import DPArgsValidator
from fedbiomed.common.exceptions import FedbiomedError


class DPController:
    """Controls DP action during training"""

    def __init__(self, training_plan: Any, dp_args: Union[Dict, None]):
        """Constructs DPController with given model

        Args:
            training_plan: An instance of torch [`Module`][torch.nn.Module]
            dp_args: Arguments for differential privacy
        """

        self._privacy_engine = PrivacyEngine()
        self._tp = training_plan
        self._dp_args = dp_args
        self._is_active = True if dp_args is not None else False

        # Configure/validate dp arguments
        if self._is_active:
            self._configure_dp_args()

    def before_training(self) -> None:
        """DP action before starting training"""
        if self._is_active:
            self._validate_and_fix_model()
            self._make_private()

    def after_training(self, params: Dict) -> Dict:
        """DP actions after the training

        Args:
            params: Contains model parameters after training with differential privacy

        Returns:
            `params` fixed model parameters after applying differential privacy
        """
        if self._is_active:
            params = self._postprocess_dp(params)

        return params

    def _make_private(self) -> None:
        """Makes model, optimizer and training data loader private

        This method directly modifies training plan attributes
        """
        self._tp._model, self._tp._optimizer, self._tp.training_data_loader = \
            self._privacy_engine.make_private(module=self._tp.model(),
                                              optimizer=self._tp.optimizer(),
                                              data_loader=self._tp.training_data_loader,
                                              noise_multiplier=self._dp_args.get('sigma'),
                                              max_grad_norm=self._dp_args.get('clip'))

    def _configure_dp_args(self):
        """Initialize arguments to perform DP training. """

        self._dp_args = DPArgsValidator.populate_with_defaults(self._dp_args, only_required=False)
        try:
            DPArgsValidator.validate(self._dp_args)
        except ValidateError as e:
            raise FedbiomedError(f"DP arguments are not valid: {e}")

        if self._dp_args['type'] == 'central':
            self._dp_args.update(sigma_CDP=self._dp_args['sigma'])
            self._dp_args['sigma'] = 0.

    def _validate_and_fix_model(self):
        """ Validate and Fix model to be DP-compliant """
        model = self._tp.model()
        if not ModuleValidator.is_valid(model):
            self._tp._model = ModuleValidator.fix(model)

    def _assess_budget_locally(self):
        """ """
        # To be used by the nodes to assess budget locally
        eps, alpha = self.privacy_engine.accountant.get_privacy_spent(delta=.1 / len(self._tp.training_data_loader))

        return eps, alpha

    def _postprocess_dp(self) -> dict:
        """Postprocess of model's parameters after training with DP.

        Postprocess of DP parameters implies:
            - If central DP is enabled, model's parameters are perturbed  according to the provided DP parameters
            - When the Opacus `PrivacyEngine` is attached to the model, parameters' names are modified by the
                addition of `_module.`. This modification should be undone before communicating to the master
                for aggregation. This is needed in order to correctly perform download/upload of model's parameters
                in the following rounds

        Returns:
            Contains (post processed) parameters
        """

        params = self._tp.model().state_dict()

        if self._dp_args['type'] == 'central':
            sigma_CDP = deepcopy(self._dp_args['sigma_CDP'])
            delta_params = {}
            perturbed_params = {}
            for name, param in params.items():
                # Extracting the update
                delta_theta = deepcopy(param)
                delta_params[name] = delta_theta - self._tp.initial_params()[name]

            for key, delta_param in delta_params.items():
                # Perturb update and update parameters
                delta_theta_tilde = deepcopy(delta_param)
                delta_theta_tilde += sigma_CDP*self._dp_args['clip'] * torch.randn_like(delta_theta_tilde)
                perturbed_params[key]= delta_theta_tilde + self._tp.initial_params()[key]
            params = deepcopy(perturbed_params)

        params_keys = list(params.keys())
        for key in params_keys:
            if '_module' in key:
                newkey = key.replace('_module.', '')
                params[newkey] = params.pop(key)

        return params
