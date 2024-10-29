# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Utils to flatten and unflatten AuxVar instances for secure aggregation."""

from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
from declearn.optimizer.modules import AuxVar
from declearn.model.api import Vector, VectorSpec
from declearn.utils import access_registered, access_registration_info
from typing_extensions import Self


__all__ = [
    "EncryptedAuxVar",
    "flatten_auxvar_for_secagg",
    "unflatten_auxvar_after_secagg",
]


ArraySpec = Tuple[List[int], str]
ValueSpec = List[Tuple[str, int, Union[None, ArraySpec, VectorSpec]]]

AuxVarT = TypeVar("AuxVarT", bound=AuxVar)


def flatten_auxvar_for_secagg(
    aux_var: Dict[str, AuxVar],
) -> Tuple[
    List[float],
    List[ValueSpec],
    List[Optional[Dict[str, Any]]],
    List[Tuple[str, Type[AuxVar]]],
]:
    """Flatten a node's optimizer auxiliary variables for secure aggregation.

    Args:
        aux_var: Optimizer auxiliary variables that are meant to be encrypted,
            formatted as a `{module_name: module_aux_var}` dict.

    Returns:
        cryptable: List of flattened encryptable values.
        enc_specs: List of module-wise specifications describing the
            flattened values' initial names, types and shapes.
        cleartext: List of module-wise optional cleartext values, that
            need sharing and aggregation but not encryption.
        clear_cls: List of module-wise tuples storing module name and
            source `AuxVar` subtype.

    Raises:
        NotImplementedError: if a module is not compatible with SecAgg.
    """
    # Iteratively flatten and gather specs from module-wise AuxVar objects.
    flattened = []  # type: List[float]
    enc_specs = []  # type: List[ValueSpec]
    cleartext = []  # type: List[Optional[Dict[str, Any]]]
    clear_cls = []  # type: List[Tuple[str, Type[AuxVar]]]
    for module_name, module_auxv in aux_var.items():
        flat, spec, clrt = _flatten_aux_var(module_auxv)
        flattened.extend(flat)
        enc_specs.append(spec)
        cleartext.append(clrt)
        clear_cls.append((module_name, type(module_auxv)))
    # Wrap up the results into an EncryptedAuxVar instance.
    return flattened, enc_specs, cleartext, clear_cls


def _flatten_aux_var(
    aux_var: AuxVar,
) -> Tuple[List[float], ValueSpec, Optional[Dict[str, Any]]]:
    """Flatten a given AuxVar instance for its encryption.

    Args:
        aux_var: `AuxVar` instance that needs encryption.

    Returns:
        flattened: flat list of float values that need encryption.
        enc_specs: list of specifications associated with flattened values.
        cleartext: optional dict of fields that are to remain cleartext.

    Raises:
        NotImplementedError: if the input is not compatible with SecAgg.
    """
    # Gather fields that need encryption and (opt.) cleartext ones.
    cryptable, cleartext = aux_var.prepare_for_secagg()
    # Iteratively flatten and fields that need encryption and build specs.
    flattened = []  # type: List[float]
    flat_spec = []  # type: ValueSpec
    for field, value in cryptable.items():
        if isinstance(value, (int, float)):
            flat = [float(value)]
            spec = None  # type: Union[None, ArraySpec, VectorSpec]
        elif isinstance(value, np.ndarray):
            flat = [float(x) for x in value.ravel().tolist()]
            spec = (list(value.shape), value.dtype.name)
        elif isinstance(value, Vector):
            flat, spec = value.flatten()
        flattened.extend(flat)
        flat_spec.append((field, len(flat), spec))
    # Return flattened encryptable values, their specs, and cleartext ones.
    return flattened, flat_spec, cleartext


def unflatten_auxvar_after_secagg(
    decrypted: List[float],
    enc_specs: List[ValueSpec],
    cleartext: List[Optional[Dict[str, Any]]],
    clear_cls: List[Tuple[str, Type[AuxVar]]],
) -> Dict[str, AuxVar]:
    """Unflatten secure-aggregate optimizer auxiliary variables.

    Args:
        decrypted: List of flattened decrypted (encryptable) values.
        enc_specs: List of module-wise specifications describing the
            flattened values' initial names, types and shapes.
        cleartext: List of module-wise optional cleartext values, that
            need sharing and aggregation but not encryption.
        clear_cls: List of module-wise tuples storing module name and
            source `AuxVar` subtype.

    Returns:
        Unflattened optimizer auxiliary variables, as a dict
        with format `{module_name: module_aux_var}`.

    Raises:
        RuntimeError: if auxiliary variables unflattening fails.
    """
    # Iteratively rebuild AuxVar instances, then return.
    aux_var = {}  # type: Dict[str, AuxVar]
    indx = 0
    for i, (name, aux_cls) in enumerate(clear_cls):
        size = sum(size for _, size, _ in enc_specs[i])
        aux_var[name] = _unflatten_aux_var(
            aux_cls=aux_cls,
            flattened=decrypted[indx:indx+size],
            enc_specs=enc_specs[i],
            cleartext=cleartext[i],
        )
    return aux_var


def _unflatten_aux_var(
    aux_cls: Type[AuxVarT],
    flattened: List[float],
    enc_specs: ValueSpec,
    cleartext: Optional[Dict[str, Any]],
) -> AuxVarT:
    """Unflatten a given AuxVar instance (resulting from SecAgg).

    Args:
        aux_cls: `AuxVar` subclass type that needs instantiation.
        flattened: flat list of (decrypted) float values.
        enc_specs: list of specifications associated with flattened values.
        cleartext: optional dict of fields that are to remain cleartext.

    Returns:
        aux_var: `aux_cls` instance that needs encryption.

    Raises:
        RuntimeError: If provided specs cannot be properly parsed.
    """
    fields = {}  # type: Dict[str, Any]
    # Iteratively rebuild flattened fields to scalar, arrays or Vector.
    indx = 0
    for name, size, spec in enc_specs:
        if (spec is None) and (size == 1):
            fields[name] = flattened[indx]
        elif isinstance(spec, (tuple, list)):
            shape, dtype = spec
            flat = flattened[indx:indx + size]
            fields[name] = np.array(flat).astype(dtype).reshape(shape)
        elif isinstance(spec, VectorSpec):
            flat = flattened[indx:indx + size]
            fields[name] = Vector.build_from_specs(flat, spec)
        else:
            raise RuntimeError(
                "Invalid specifications for flattened auxiliary variables."
            )
        indx += size
    # Try instantiating from rebuilt and preserved fields.
    if cleartext:
        fields.update(cleartext)
    try:
        return aux_cls.from_dict(fields)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to instantiate '{aux_cls}' instance from flattened "
            "values, in spite of fields' unflattening having gone well."
        ) from exc


class EncryptedAuxVar:
    """Container for encrypted optimizer auxiliary variables.

    This _ad hoc_ data structure is designed to enable performing secure
    aggregation over auxiliary variables of declearn-backed optimizers.

    It is designed to be used in four steps:

    - Encrypt the outputs of a declearn optimizer's `collect_aux_var` call,
      using `flatten_auxvar_for_secagg` and a `SecaggCrypter`, then wrap the
      results into an `EncryptedAuxVar`
    - Convert the `EncryptedAuxVar` to and from a serializable dict, enabling
      to transmit it across network communications (from nodes to researcher).
    - Aggregate node-wise encrypted values by summing nodes' `EncryptedAuxVar`
      instances (or calling directly their `aggregate` method).
    - Decrypt the resulting instance's `encrypted` values with a `SecaggCrypter`
      and use `unflatten_auxvar_after_secagg` on the decrypted values and the
      rest of the instance's attributes to recover auxiliary variables that can
      be passed to the researcher's optimizer's `process_aux_var` method.
    """

    def __init__(
        self,
        encrypted: List[List[int]],
        enc_specs: List[ValueSpec],
        cleartext: List[Optional[Dict[str, Any]]],
        clear_cls: List[Tuple[str, Type[AuxVar]]],
    ) -> None:
        """Instantiate a container from already encrypted information.

        Args:
            encrypted: List of node-wise flattened, encrypted values.
            enc_specs: List of module-wise specifications describing the
                flattened values' initial names, types and shapes.
            cleartext: List of module-wise optional cleartext values, that
                need sharing and aggregation but not encryption.
            clear_cls: List of module-wise tuples storing module name and
                source `AuxVar` subtype.
        """
        self.encrypted = encrypted
        self.enc_specs = enc_specs
        self.cleartext = cleartext
        self.clear_cls = clear_cls

    def get_num_expected_params(
        self,
    ) -> int:
        """Return the number of flat values that should be decrypted."""
        return sum(
            size
            for module_specs in self.enc_specs
            for _, size, _ in module_specs
        )

    def concatenate(
        self,
        other: Self,
    ) -> Self:
        """Concatenates a pair of EncryptedAuxVar into a single instance.

        Args:
            other: `EncryptedAuxVar` instance to be aggregated with this one.

        Returns:
            `EncryptedAuxVar` instance resulting from the aggregation (with
            concatenated encrypted values, and aggregated cleartext ones).

        Raises:
            TypeError: if `other` is not an `EncryptedAuxVar` instance.
            ValueError: if `other` has distinct specs from this one.
        """
        # Raise if instances do not match.
        if not isinstance(other, self.__class__):
            raise TypeError(
                f"'{self.__class__.__name__}.aggregate' expects an input "
                f"with the same type, but received '{type(other)}'."
            )
        if self.enc_specs != other.enc_specs:
            raise ValueError(
                f"Cannot sum '{self.__class__.__name__}' instances with"
                " distinct specs for encrypted values."
            )
        if self.clear_cls != other.clear_cls:
            raise ValueError(
                f"Cannot sum '{self.__class__.__name__}' instances with"
                " distinct specs for base AuxVar classes."
            )
        # Concatenate lists of encrypted values for future sum-decryption.
        encrypted = self.encrypted + other.encrypted
        # Perform aggregation of cleartext values, using type-specific rules.
        cleartext = [
            self._aggregate_cleartext(
                aux_cls, self.cleartext[i], other.cleartext[i]
            )
            for i, (_, aux_cls) in enumerate(self.clear_cls)
        ]
        # Wrap up results in an EncryptedAuxVar instance and return it.
        return self.__class__(
            encrypted=encrypted,
            enc_specs=self.enc_specs,
            cleartext=cleartext,
            clear_cls=self.clear_cls,
        )

    @staticmethod
    def _aggregate_cleartext(
        aux_cls: Type[AuxVar],
        clr_txt_a: Optional[Dict[str, Any]],
        clr_txt_b: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Aggregate cleartext fields of two instances of an AuxVar class."""
        if (clr_txt_a is None) or (clr_txt_b is None):
            return None
        default = aux_cls.default_aggregate
        outputs = {}  # type: Dict[str, Any]
        for key, val_a in clr_txt_a.items():
            val_b = clr_txt_b[key]
            agg_f = getattr(aux_cls, f"aggregate_{key}", default)
            outputs[key] = agg_f(val_a, val_b)
        return outputs

    def __add__(
        self,
        other: Self,
    ) -> Self:
        """Overload addition operator to call `self.concatenate`."""
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.concatenate(other)

    @classmethod
    def concatenate_from_dict(cls, data: Dict[str, Self]) -> Self:
        auxvar = list(data.values())
        # this converts List[List[List[int]]] -> List[List[int]]
        obj = sum(auxvar[1:], start=auxvar[0])
        return cls(obj.encrypted, obj.enc_specs, obj.cleartext, obj.clear_cls)

    def get_mapping_encrypted_aux_var(self) -> Dict[str, List[int]]:
        nodes_id = list(self.cleartext[0]['clients'])
        return {n: p for n,p in zip(nodes_id, self.encrypted)}

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        """Return a dict representation of this instance.

        Returns:
            Dict representation of this instance.
        """
        aux_cls_info = [
            (name, access_registration_info(aux_cls))
            for name, aux_cls in self.clear_cls
        ]
        return {
            "encrypted": self.encrypted,
            "enc_specs": self.enc_specs,
            "cleartext": self.cleartext,
            "clear_cls": aux_cls_info,
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
    ) -> Self:
        """Instantiate from a dict representation.

        Args:
            data: Dict representation, as emitted by this class's `to_dict`.

        Raises:
            TypeError: If any required key is missing or improper.
        """
        try:
            # Recover wrapped AuxVar classes from the type registry.
            clear_cls = [
                (name, access_registered(*info))
                for name, info in data["clear_cls"]
            ]
            # Ensure tuples are preserved (as serialization converts to list).
            enc_specs = [
                [tuple(value_specs) for value_specs in module_specs]
                for module_specs in data["enc_specs"]
            ]
            # Try instantiating from the input data.
            return cls(
                encrypted=data["encrypted"],
                enc_specs=enc_specs,  # type: ignore
                cleartext=data["cleartext"],
                clear_cls=clear_cls,
            )
        except Exception as exc:
            raise TypeError(
                f"Cannot instantiate '{cls.__name__}' from input dict: "
                f"raised '{repr(exc)}'."
            ) from exc
