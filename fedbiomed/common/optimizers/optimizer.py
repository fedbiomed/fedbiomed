# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Optimizer class wrapping the declearn-issued Optimizer."""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from declearn.model.api import Vector
from declearn.optimizer import Optimizer as DeclearnOptimizer
from declearn.optimizer.modules import AuxVar, OptiModule
from declearn.optimizer.regularizers import Regularizer
from typing_extensions import Self

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedOptimizerError
from fedbiomed.common.optimizers.declearn import set_device_policy


class Optimizer:
    """Optimizer class with a declearn-backed modular SGD-core algorithm."""

    def __init__(
        self,
        lr: float,
        decay: float = 0.0,
        modules: Optional[
            Sequence[Union[OptiModule, str, Tuple[str, Dict[str, Any]]]]
        ] = None,
        regularizers: Optional[
            Sequence[Union[Regularizer, str, Tuple[str, Dict[str, Any]]]]
        ] = None,
    ) -> None:
        """Instantiate the declearn-issued gradient-descent optimizer.

        Args:
            lr: Base learning rate (i.e. step size) applied to gradients-based
                updates upon applying them to a model's weights.
            decay: Optional weight decay parameter, used to parameterize a
                decoupled weight decay regularization term (see [1]) added to
                the updates right before the learning rate is applied and model
                weights are effectively updated.
            modules: Optional list of plug-in modules implementing gradients'
                alteration into model weights' updates. Modules will be applied
                to gradients following this list's ordering.
                See `declearn.optimizer.modules.OptiModule` for details.
                See Notes section below for details on the "specs" format.
            regularizers: Optional list of plug-in loss regularizers.
                Regularizers will be applied to gradients following this list's
                order, prior to any other alteration (see `modules` above).
                See `declearn.optimizer.regularizers.Regularizer` for details.
                See Notes section below for details on the "specs" format.

        !!! info "Note"
            `Regularizer` and `OptiModule` to be used by this optimizer,
            specified using the `regularizers` and `modules` parameters,
            may be passed as ready-for-use instances, or be instantiated
            from specs, consisting either of a single string (the `name`
            attribute of the class to build) or a tuple grouping this
            name and a config dict (to specify some hyper-parameters).

        !!! info "References"
            [1] Loshchilov & Hutter, 2019.
                Decoupled Weight Decay Regularization.
                https://arxiv.org/abs/1711.05101
        """
        try:
            self._optimizer = DeclearnOptimizer(
                lrate=lr,
                w_decay=decay,
                modules=modules,
                regularizers=regularizers,
            )
        except (KeyError, TypeError) as exc:
            raise FedbiomedOptimizerError(
                f"{ErrorNumbers.FB621.value}: declearn Optimizer instantiation"
                f" raised the following exception: {repr(exc)}"
            ) from exc

    @classmethod
    def from_declearn_optimizer(
        cls,
        declearn_optimizer: DeclearnOptimizer,
    ) -> Self:
        """Wrap a declearn Optimizer into a fed-biomed one.

        Args:
            declearn_optimizer: [declearn.optimizer.Optimizer][] instance that
                needs to be wrapped.

        Returns:
            Fed-BioMed `Optimizer` instance wrapping a copy of the input
                declearn optimizer.
        """
        config = declearn_optimizer.get_config()
        optim = cls(
            lr=config["lrate"],
            decay=config["w_decay"],
            modules=config["modules"],
            regularizers=config["regularizers"],
        )
        optim._optimizer.set_state(declearn_optimizer.get_state())
        return optim

    def init_round(self) -> None:
        """Trigger start-of-training-round behavior of wrapped regularizers."""
        try:
            self._optimizer.start_round()
        except Exception as exc:
            raise FedbiomedOptimizerError(
                f"{ErrorNumbers.FB621.value}: error in 'init_round': {exc}"
            ) from exc

    def step(self, grads: Vector, weights: Vector) -> Vector:
        """Run an optimization step to compute and return model weight updates.

        Use the pre-assigned `weights` and `grads` (set using the `set_weights`
        and `set_grads` methods) to compute weight updates, using the pipeline
        defined by this instance.

        Args:
            grads: Raw gradients based on which to compute weights updates,
                wrapped into a declearn Vector structure.
            weights: Current values of the weights with respect to which the
                gradients were computed, wrapped into a declearn Vector with
                the same concrete type as `grads`.

        Returns:
            Updates to be applied to the model weights, computed by:
                - running wrapped gradients and weights through the regularizer
                  plug-ins (that add loss-regularization terms' derivatives);
                - running resulting gradients through the optimodule plug-ins
                  (that perform any defined gradient-alteration operation);
                - adding a decoupled weight-decay term, if one is to be used;
                - scaling the updates by the base learning rate.
                The results are wrapped into a declearn Vector structure, the
                concrete type of which is same as input `grads` and `weights`.
        """
        # This code mostly replicates that of
        # `declearn.optimizer.Optimizer.compute_updates_from_gradients`.
        try:
            # Add loss-regularization terms' derivatives to the raw gradients.
            for reg in self._optimizer.regularizers:
                grads = reg.run(grads, weights)
            # Iteratively refine updates by running them through the optimodules.
            for mod in self._optimizer.modules:
                grads = mod.run(grads)
            # Apply the base learning rate.
            updates = -self._optimizer.lrate * grads
            # Optionally add the decoupled weight decay term.
            if self._optimizer.w_decay:
                updates -= self._optimizer.w_decay * weights
            # Return the model updates.
            return updates
        except Exception as exc:
            raise FedbiomedOptimizerError(
                f"{ErrorNumbers.FB621.value}: error in 'step': {exc}"
            ) from exc

    def get_aux(self) -> Dict[str, AuxVar]:
        """Return auxiliary variables that need to be shared across network.

        Returns:
            Aux-var dict that associates `module.collect_aux_var()` values to
                `module.name` keys for each and every module plugged in this
                Optimizer that has some auxiliary variables to share.

        !!! info "Note"
            "Auxiliary variables" are information that needs to be shared
            between the nodes and the researcher between training rounds, to
            synchronize some optimizer plug-ins that work by pair. Their
            production via this method can have internal side effects;
            `get_aux` should therefore be called sparingly.
        """
        try:
            return self._optimizer.collect_aux_var()
        except Exception as exc:
            raise FedbiomedOptimizerError(
                f"{ErrorNumbers.FB621.value}: error in 'get_aux': {exc}"
            ) from exc

    def set_aux(self, aux: Dict[str, AuxVar]) -> None:
        """Update plug-in modules based on received shared auxiliary variables.

        Args:
            aux: Auxiliary variables received from the counterpart optimizer
                (on the other side of the node-researcher frontier). On the
                researcher side, values must have been pre-aggregated based
                on the ones sent by nodes.

        Raises:
            FedbiomedOptimizerError: If a key from `aux_var` does not match the
                name of any module plugged in this optimizer (i.e. if received
                variables cannot be mapped to a destinatory module).

        !!! info "Note"
            "Auxiliary variables" are information that is shared between the
            nodes and researcher between training rounds, to synchronize some
            optimizer plug-ins that work by pair. The inputs to this method are
            not simply stored by the Optimizer, but are processed into internal
            side effects; this method should therefore be called sparingly.
        """
        try:
            self._optimizer.process_aux_var(aux)
        except Exception as exc:
            raise FedbiomedOptimizerError(
                f"{ErrorNumbers.FB621.value}: `Optimizer.set_aux`: {exc}"
            ) from exc

    def filter_aux(
        self, aux_vars: Dict[str, AuxVar], private_params: list[str]
    ) -> Dict[str, AuxVar]:
        """Remove selected parameter entries from Declearn auxiliary variables.

        Iterates over optimizer auxiliary variables collected from Declearn
        modules and removes the specified parameter names from any contained
        Vector objects (e.g. TorchVector, NumpyVector). The filtering is applied
        to every module present in the auxiliary variable dictionary.

        Args:
            aux_vars: Dictionary of auxiliary variables,
                mapping module names to their corresponding ``AuxVar`` objects.
            private_params: List of parameter names to remove from auxiliary
                vectors (e.g. ``"conv1.weight"``).

        Returns:
            The modified auxiliary variables dictionary with the selected
            parameters removed from any contained vectors.

        Raises:
            FedbiomedOptimizerError: If ``aux_vars`` is not a dictionary, if an
                unexpected vector structure is encountered, or if an error occurs
                while processing auxiliary variables for a module.
        """

        if not isinstance(aux_vars, dict):
            raise FedbiomedOptimizerError(
                f"{ErrorNumbers.FB621.value}: Auxiliary variables must be a dict"
            )

        for module_name, auxvar in aux_vars.items():
            try:
                # Iterate through AuxVar attributes
                for attr_name in vars(auxvar):
                    attr_val = getattr(auxvar, attr_name)

                    # Identify Declearn vectors by presence of `.coefs`
                    if hasattr(attr_val, "coefs"):
                        vector = attr_val

                        if not isinstance(vector.coefs, dict):
                            raise FedbiomedOptimizerError(
                                f"{ErrorNumbers.FB621.value}: Unexpected vector structure in module '{module_name}', "
                                f"attribute '{attr_name}'"
                            )

                        for layer in private_params:
                            vector.coefs.pop(layer, None)

            except Exception as exc:
                raise FedbiomedOptimizerError(
                    f"{ErrorNumbers.FB621.value}: Failed while processing auxiliary variables for module "
                    f"'{module_name}': {exc}"
                ) from exc

        return aux_vars

    def restore_aux(
        self,
        aux_vars: Dict[str, AuxVar],
        reference_params: Dict[str, Any],
        private_params: list[str],
    ) -> Dict[str, AuxVar]:
        """Restore missing auxiliary variables for private model parameters.

        Ensures that auxiliary variables received from the researcher contain
        entries for parameters that are kept private on the node side. For each
        missing parameter name, a zero-valued tensor matching the local model
        parameter shape is inserted into the corresponding Declearn Vector.

        Args:
            aux_vars: Dictionary of auxiliary variables received from the
                researcher.
            reference_vector: local model weights providing reference tensors
                for model parameters (used to infer tensor shapes and dtypes).
            private_params: List of parameter names that are private to the node
                and therefore absent from researcher auxiliary variables
                (e.g. ``"conv1.weight"``).

        Returns:
            The modified auxiliary variables dictionary where missing private
            parameters have been restored with zero-valued tensors.

        Raises:
            FedbiomedOptimizerError: If ``aux_vars`` is not a dictionary, if
                reference tensors cannot be mapped to Declearn Vector, if
                reference tensors cannot be found for private parameters, or
                if an unexpected auxiliary-variable structure is encountered.
        """

        if not isinstance(aux_vars, dict):
            raise FedbiomedOptimizerError(
                f"{ErrorNumbers.FB621.value}: Auxiliary variables must be a dict"
            )

        # Try to convert reference parameters to Declearn Vector
        try:
            reference_vector = Vector.build(reference_params)
        except Exception as exc:
            raise FedbiomedOptimizerError(
                f"{ErrorNumbers.FB621.value}: Failed while instantiating Declearn "
                f"Vector: {exc}"
            ) from exc

        for module_name, auxvar in aux_vars.items():
            try:
                for attr_name in vars(auxvar):
                    attr_val = getattr(auxvar, attr_name)

                    if hasattr(attr_val, "coefs"):
                        vector = attr_val

                        if not isinstance(vector.coefs, dict):
                            raise FedbiomedOptimizerError(
                                f"{ErrorNumbers.FB621.value}: Unexpected vector structure "
                                f"in module '{module_name}', attribute '{attr_name}'"
                            )

                        for layer in private_params:
                            # If layer missing, recreate zero tensor
                            if layer not in vector.coefs:
                                if layer not in reference_vector.coefs:
                                    raise FedbiomedOptimizerError(
                                        f"{ErrorNumbers.FB621.value}: Cannot infer shape "
                                        f"for private parameter '{layer}'"
                                    )

                                ref_tensor = reference_vector.coefs[layer]
                                vector.coefs[layer] = ref_tensor.clone().zero_()

            except Exception as exc:
                raise FedbiomedOptimizerError(
                    f"{ErrorNumbers.FB621.value}: Failed while restoring auxiliary "
                    f"variables for module '{module_name}': {exc}"
                ) from exc

        return aux_vars

    def get_aux_names(self) -> List[str]:
        """Gathers list of names of modules requiring auxiliary variables"""
        aux_names = []

        for module in self._optimizer.modules:
            if module.aux_name is not None:
                aux_names.append(module.aux_name)
        return aux_names

    def get_state(self) -> Dict[str, Any]:
        """Return the configuration and current states of this Optimizer.

        This method is to be used for creating breakpoints.

        Returns:
            State-and-config dict that may be saved as part of a breakpoint
                file, and used to re-create this Optimizer using the
                `Optimizer.load_state` classmethod constructor.
        """
        try:
            config = self._optimizer.get_config()
            states = self._optimizer.get_state()
            return {"config": config, "states": states}
        except Exception as exc:
            raise FedbiomedOptimizerError(
                f"{ErrorNumbers.FB621.value}: error in 'get_state': {exc}"
            ) from exc

    @classmethod
    def load_state(cls, state: Dict[str, Any]) -> Self:
        """Instantiate an Optimizer from its breakpoint state dict.

        Args:
            state: state-and-config dict created using the `get_state` method.

        Returns:
            Optimizer instance re-created from the `state` dict.

        Raises:
            FedbiomedOptimizerError: If the input `state` dict has improper keys
                or fails to set up a declearn Optimizer and set back its state.
        """
        try:
            optim = DeclearnOptimizer.from_config(state["config"])
            optim.set_state(state["states"])
        except KeyError as exc:
            raise FedbiomedOptimizerError(
                f"{ErrorNumbers.FB621.value}: Missing field in the breakpoints state: {exc}"
            ) from exc
        except Exception as exc:
            raise FedbiomedOptimizerError(
                f"{ErrorNumbers.FB621.value}: `Optimizer.load_state`: {exc}"
            ) from exc
        return cls(
            lr=optim.lrate,
            decay=optim.w_decay,
            modules=optim.modules,
            regularizers=optim.regularizers,
        )

    def send_to_device(self, device: Union[str, bool], idx: Optional[int] = None):
        """GPU support"""
        # for now GPU support on Researcher side is disabled
        set_device_policy(device, idx)
