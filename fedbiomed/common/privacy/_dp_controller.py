from torch import randn_like
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Union, Dict, Tuple
from copy import deepcopy
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from fedbiomed.common.validator import ValidateError
from fedbiomed.common.training_args import DPArgsValidator
from fedbiomed.common.exceptions import FedbiomedDPControllerError
from fedbiomed.common.constants import ErrorNumbers


class DPController:
    """Controls DP action during training"""

    def __init__(self, dp_args: Union[Dict, None] = None):
        """Constructs DPController with given model.

        Args:
            dp_args: Arguments for differential privacy
        """

        self._privacy_engine = PrivacyEngine()
        self._dp_args = dp_args
        self._is_active = True if dp_args is not None else False

        # Configure/validate dp arguments
        if self._is_active:
            self._configure_dp_args()

    def before_training(self,
                        model: Module,
                        optimizer: Optimizer,
                        loader: DataLoader) -> Tuple[Module, Optimizer, DataLoader]:
        """DP action before starting training.

        Args:
            model: Model that will be used for training
            optimizer: Optimizer for training
            loader: Data loader for training

        Returns:
            Differential privacy applies model, optimizer and data loader
        """

        if not isinstance(model, Module):
            raise FedbiomedDPControllerError(f"{ErrorNumbers.FB616}: Model must be an instance of torch.nn.Module")

        if not isinstance(optimizer, Optimizer):
            raise FedbiomedDPControllerError(f"{ErrorNumbers.FB616}: Optimizer must be an instance of "
                                             f"torch.optim.Optimizer")

        if not isinstance(loader, DataLoader):
            raise FedbiomedDPControllerError(f"{ErrorNumbers.FB616}: Data loader must be an instance of "
                                             f"torch.utils.data.DataLoader")

        if self._is_active:
            model = self._validate_and_fix_model(model)
            try:
                model, optimizer, loader = \
                    self._privacy_engine.make_private(module=model,
                                                      optimizer=optimizer,
                                                      data_loader=loader,
                                                      noise_multiplier=self._dp_args.get('sigma'),
                                                      max_grad_norm=self._dp_args.get('clip'))
            except Exception as e:
                raise FedbiomedDPControllerError(f"{ErrorNumbers.FB616.value}: Error while running privacy "
                                                 f"engine: {e}")

        return model, optimizer, loader

    def after_training(self, params: Dict, initial_params: Dict) -> Dict:
        """DP actions after the training.

        Args:
            params: Contains model parameters after training with differential privacy
            initial_params: Initial parameters before training with  differential privacy
        Returns:
            `params` fixed model parameters after applying differential privacy
        """
        if self._is_active:
            params = self._postprocess_dp(params, initial_params)

        return params

    def _configure_dp_args(self):
        """Initialize arguments to perform DP training. """

        self._dp_args = DPArgsValidator.populate_with_defaults(self._dp_args, only_required=False)
        try:
            DPArgsValidator.validate(self._dp_args)
        except ValidateError as e:
            raise FedbiomedDPControllerError(f"{ErrorNumbers.FB616.value}: DP arguments are not valid: {e}")

        if self._dp_args['type'] == 'central':
            self._dp_args.update(sigma_CDP=self._dp_args['sigma'])
            self._dp_args['sigma'] = 0.

    @staticmethod
    def _validate_and_fix_model(model: Module) -> Module:
        """Validate and Fix model to be DP-compliant.

        Args:
            model: An instance of [`Module`][torch.nn.Module]

        Returns:
            Fixed or validated model
        """
        if not ModuleValidator.is_valid(model):
            try:
                model = ModuleValidator.fix(model)
            except Exception as e:
                raise FedbiomedDPControllerError(f"{ErrorNumbers.FB616.value}: Error while making model DP-compliant"
                                                 f"{e}")

        return model

    def _assess_budget_locally(self, loader) -> Tuple[float, float]:
        """Computes eps and alpha for budget privacy.

        TODO: This function is not used any where on the node side.

        Args:
            loader: Pytorch data loader that is going to be used for training

        Returns:
            eps: Calculated epsilon value for privacy budget
            alpha: Calculated epsilon alpha for privacy budget
        """
        # To be used by the nodes to assess budget locally
        eps, alpha = self._privacy_engine.accountant.get_privacy_spent(delta=.1 / len(loader))

        return eps, alpha

    def _postprocess_dp(self, params: Dict, initial_params: Dict) -> Dict:
        """Postprocess of model's parameters after training with DP.

        **Postprocess of DP parameters implies**
        - If central DP is enabled, model's parameters are perturbed  according to the provided DP parameters
        - When the Opacus `PrivacyEngine` is attached to the model, parameters' names are modified by the
            addition of `_module.`. This modification should be undone before communicating to the master
            for aggregation. This is needed in order to correctly perform download/upload of model's parameters
            in the following rounds

        Args:
            params: Contains model parameters after training with differential privacy
            initial_params: Initial parameters before training with  differential privacy

        Returns:
            Contains (post processed) parameters
        """

        if self._dp_args['type'] == 'central':
            sigma_CDP = deepcopy(self._dp_args['sigma_CDP'])
            delta_params = {}
            perturbed_params = {}
            for name, param in params.items():
                # Extracting the update
                delta_theta = deepcopy(param)
                delta_params[name] = delta_theta - initial_params[name]

            for key, delta_param in delta_params.items():
                # Perturb update and update parameters
                delta_theta_tilde = deepcopy(delta_param)
                delta_theta_tilde += sigma_CDP * self._dp_args['clip'] * randn_like(delta_theta_tilde)
                perturbed_params[key] = delta_theta_tilde + initial_params[key]
            params = deepcopy(perturbed_params)

        params_keys = list(params.keys())
        for key in params_keys:
            if '_module' in key:
                newkey = key.replace('_module.', '')
                params[newkey] = params.pop(key)

        return params
