"""Tests for node-side FedComBat jobs."""

from pathlib import Path

import polars as pl
import pytest
import torch

from fedbiomed.common.constants import HarmonizationStep
from fedbiomed.node.jobs import _fedcombat_jobs as module
from fedbiomed.node.jobs._fedcombat_jobs import _FedCombatJobs


class DummyDatasetManager:
    def __init__(self):
        self.deleted = []
        self.last_add_kwargs = {}

    def delete_dataset_by_id(self, dataset_id):
        self.deleted.append(dataset_id)

    def add_dynamic_dataset(self, **kwargs):
        self.last_add_kwargs = kwargs
        return "dyn_ds_1"

    def get_dataset_entry_by_id(self, dataset_id):
        return (
            {
                "dataset_id": dataset_id,
                "path": self.last_add_kwargs.get("path", ""),
                "name": "dynamic",
                "description": "dynamic-desc",
                "private": "secret",
            },
            None,
        )

    def obfuscate_private_information(self, entries):
        for entry in entries:
            entry.pop("private", None)


class DummyNodeStateManager:
    def __init__(self, state_path):
        self._previous_state_id = "prev_state"
        self._state_path = str(state_path)
        self.add_calls = []
        self.get_return = {"preproc_state": {"preproc_params_path": self._state_path}}

    def generate_folder_and_create_file_name(self, _exp_id, _step, _name):
        return self._state_path

    def add(self, experiment_id, state):
        self.add_calls.append((experiment_id, state))

    def get(self, _experiment_id, _state_id):
        return self.get_return


class DummySerializer:
    store = {}

    @staticmethod
    def dump(values, path):
        DummySerializer.store[path] = values

    @staticmethod
    def load(path):
        return DummySerializer.store[path]


@pytest.fixture
def fedcombat_jobs(tmp_path):
    dataset_manager = DummyDatasetManager()
    state_manager = DummyNodeStateManager(tmp_path / "preproc_state.msgpack")
    dataset_entry = {
        "dataset_id": "ds1",
        "path": str(tmp_path / "source.csv"),
        "name": "source-name",
        "description": "source-desc",
        "dataset_parameters": {"a": 1},
    }

    return _FedCombatJobs(
        root_dir=str(tmp_path),
        experiment_id="exp1",
        preproc_id="pre1",
        researcher_id="res1",
        node_state_manager=state_manager,  # type: ignore[arg-type]
        dataset_manager=dataset_manager,  # type: ignore[arg-type]
        dataset_entry=dataset_entry,
    )


def test_call_dispatches_supported_steps(monkeypatch, fedcombat_jobs):
    monkeypatch.setattr(fedcombat_jobs, "_standardize_data", lambda p: {"s": p})
    monkeypatch.setattr(
        fedcombat_jobs,
        "_compute_residual_variance",
        lambda p: {"rv": p},
    )
    monkeypatch.setattr(
        fedcombat_jobs,
        "_compute_standardized_residuals_params",
        lambda p: {"rp": p},
    )
    monkeypatch.setattr(
        fedcombat_jobs,
        "_compute_fedcombat_params",
        lambda p: {"fc": p},
    )
    fedcombat_jobs.step_functions = {
        HarmonizationStep.STANDARDIZE: fedcombat_jobs._standardize_data,
        HarmonizationStep.TRAIN_RESID_VAR: fedcombat_jobs._compute_residual_variance,
        HarmonizationStep.RESID_PARAMS: fedcombat_jobs._compute_standardized_residuals_params,
        HarmonizationStep.FC_PARAMS: fedcombat_jobs._compute_fedcombat_params,
    }

    assert fedcombat_jobs(HarmonizationStep.STANDARDIZE, "csv", {"k": 1}) == {
        "s": {"k": 1}
    }
    assert fedcombat_jobs(HarmonizationStep.TRAIN_RESID_VAR, "csv", {"k": 2}) == {
        "rv": {"k": 2}
    }
    assert fedcombat_jobs(HarmonizationStep.RESID_PARAMS, "csv", {"k": 3}) == {
        "rp": {"k": 3}
    }
    assert fedcombat_jobs(HarmonizationStep.FC_PARAMS, "csv", {"k": 4}) == {
        "fc": {"k": 4}
    }


def test_call_rejects_non_csv(fedcombat_jobs):
    with pytest.raises(ValueError, match="not supported"):
        fedcombat_jobs(HarmonizationStep.STANDARDIZE, "parquet", {})


def test_standardize_data(monkeypatch, fedcombat_jobs):
    covariates = torch.tensor([[1.0], [3.0]])
    phenotypes = torch.tensor([[2.0], [6.0]])
    captured = {}

    monkeypatch.setattr(
        fedcombat_jobs,
        "_read_initial_dataset_values",
        lambda *_: (covariates, phenotypes),
    )
    monkeypatch.setattr(
        fedcombat_jobs,
        "_save_updated_dataset_values",
        lambda **kwargs: {"dataset_id": "std_ds", "kwargs": kwargs},
    )
    monkeypatch.setattr(
        fedcombat_jobs,
        "_save_state_values",
        lambda s: captured.setdefault("state", s),
    )
    fedcombat_jobs._harmonization_step = HarmonizationStep.STANDARDIZE

    out = fedcombat_jobs._standardize_data(
        {
            "covariates": ["cov1"],
            "phenotypes": ["phen1"],
            "global_mean_covariates": torch.tensor([2.0]),
            "global_mean_phenotypes": torch.tensor([4.0]),
            "global_std_covariates": torch.tensor([1.0]),
            "global_std_phenotypes": torch.tensor([2.0]),
        }
    )

    assert out["standardized_dataset"]["dataset_id"] == "std_ds"
    assert captured["state"]["n_samples"] == 2
    assert captured["state"]["std_dataset_id"] == "std_ds"


def test_compute_residual_variance(monkeypatch, fedcombat_jobs):
    monkeypatch.setattr(fedcombat_jobs, "_delete_temporary_dataset", lambda _id: None)
    monkeypatch.setattr(fedcombat_jobs, "_save_state_values", lambda _state: None)
    fedcombat_jobs._read_state_values = lambda: {
        "global_mean_phenotypes": torch.tensor([0.0]),
        "global_std_phenotypes": torch.tensor([1.0]),
        "covariates_name": ["cov1"],
        "phenotypes_name": ["phen1"],
        "n_samples": 2,
        "standardized_covariates": torch.tensor([[1.0], [2.0]]),
        "standardized_phenotypes": torch.tensor([[2.0], [3.0]]),
        "std_dataset_id": "tmp_std_ds",
    }

    class DummyBio:
        def __init__(self, *_, **__):
            pass

        def load_state_dict(self, _):
            return None

        def __call__(self, x):
            return x

    class DummyBias:
        def __init__(self, *_, **__):
            pass

        def load_state_dict(self, _):
            return None

        def __call__(self, x):
            return x * 0.0

    monkeypatch.setattr(module, "FedCombatBiologicalModel", DummyBio)
    monkeypatch.setattr(module, "FedCombatBiasModel", DummyBias)

    out = fedcombat_jobs._compute_residual_variance(
        {
            "biological_model": {},
            "global_bias_model": {},
            "local_bias_model": {},
        }
    )

    assert set(out.keys()) == {"residual_variance", "n_samples"}
    assert int(out["n_samples"]) == 2


def test_compute_standardized_residuals_params(monkeypatch, fedcombat_jobs):
    saved = {}
    monkeypatch.setattr(
        fedcombat_jobs,
        "_save_state_values",
        lambda s: saved.setdefault("state", s),
    )
    fedcombat_jobs._read_state_values = lambda: {
        "global_mean_phenotypes": torch.tensor([1.0]),
        "global_std_phenotypes": torch.tensor([2.0]),
        "phenotypes_name": ["phen1"],
        "n_samples": 2,
        "standardized_phenotypes": torch.tensor([[2.0], [4.0]]),
        "biological_model_values": torch.tensor([[1.0], [1.0]]),
        "global_bias_values": torch.tensor([[0.5], [0.5]]),
    }

    out = fedcombat_jobs._compute_standardized_residuals_params(
        {"sigma_hat_g": torch.tensor([1.0])}
    )

    assert set(out.keys()) == {"gamma_hat_ig", "delta_hat_ig"}
    assert "standardized_residuals" in saved["state"]


@pytest.mark.parametrize("standardize_result", [True, False])
def test_compute_fedcombat_params(monkeypatch, fedcombat_jobs, standardize_result):
    captured = {}
    monkeypatch.setattr(
        fedcombat_jobs,
        "_save_updated_dataset_values",
        lambda phenotypes_name, phenotypes_data, **kwargs: {
            "dataset_id": "harm_ds",
            "phenotypes_name": phenotypes_name,
            "data": captured.setdefault("data", phenotypes_data),
            "kwargs": kwargs,
        },
    )
    fedcombat_jobs._read_state_values = lambda: {
        "global_mean_phenotypes": torch.tensor([10.0]),
        "global_std_phenotypes": torch.tensor([2.0]),
        "phenotypes_name": ["phen1"],
        "n_samples": 3,
        "sigma_hat_g": torch.tensor([1.0]),
        "standardized_residuals": torch.tensor([[1.0], [2.0], [3.0]]),
        "biological_model_values": torch.tensor([[0.2], [0.2], [0.2]]),
        "global_bias_values": torch.tensor([[0.1], [0.1], [0.1]]),
    }

    out = fedcombat_jobs._compute_fedcombat_params(
        {
            "gamma_bar": torch.tensor([1.0]),
            "tau_2": torch.tensor([1.0]),
            "lambda_bar_i": torch.tensor([2.0]),
            "theta_bar_i": torch.tensor([1.0]),
            "standardize_result": standardize_result,
        }
    )

    assert out["harmonized_dataset"]["dataset_id"] == "harm_ds"
    assert isinstance(captured["data"], torch.Tensor)


def test_read_initial_dataset_values(monkeypatch, fedcombat_jobs):
    class DummyTabularDataset:
        def __init__(self, input_columns, target_columns):
            self.input_columns = input_columns
            self.target_columns = target_columns
            self.data = [
                (torch.tensor([1.0]), torch.tensor([10.0])),
                (torch.tensor([2.0]), torch.tensor([20.0])),
            ]

        def load(self, **_kwargs):
            return None

        def __len__(self):
            return len(self.data)

        def __getitem__(self, i):
            return self.data[i]

    monkeypatch.setattr(module, "TabularDataset", DummyTabularDataset)

    cov, phe = fedcombat_jobs._read_initial_dataset_values(["cov1"], ["phen1"])

    assert cov.shape == (2, 1)
    assert phe.shape == (2, 1)


def test_delete_temporary_dataset(monkeypatch, fedcombat_jobs):
    class DummyPath:
        unlinked_names = []

        def __init__(self, path):
            self._path = str(path)
            self.name = self._path.split("/")[-1]

        def exists(self):
            return True

        def unlink(self):
            DummyPath.unlinked_names.append(self.name)

    fedcombat_jobs._dataset_manager.last_add_kwargs["path"] = "/tmp/to_delete.csv"
    monkeypatch.setattr(module, "Path", DummyPath)

    fedcombat_jobs._delete_temporary_dataset("to_delete")

    assert fedcombat_jobs._dataset_manager.deleted == ["to_delete"]
    assert DummyPath.unlinked_names == ["to_delete.csv"]


@pytest.mark.parametrize("is_harmonization", [True, False])
@pytest.mark.parametrize("with_covariates", [True, False])
def test_save_updated_dataset_values(
    monkeypatch,
    fedcombat_jobs,
    is_harmonization,
    with_covariates,
):
    class DummyTabularController:
        def __init__(self, root):
            assert root == fedcombat_jobs._dataset_entry["path"]
            self._samples = [
                pl.DataFrame({"cov1": [0.0], "phen1": [1.0]}),
                pl.DataFrame({"cov1": [1.0], "phen1": [2.0]}),
            ]

        def __len__(self):
            return len(self._samples)

        def get_sample(self, i):
            return self._samples[i]

    monkeypatch.setattr(module, "TabularController", DummyTabularController)
    monkeypatch.setattr(pl.DataFrame, "write_csv", lambda self, path: None)

    phenotypes_data = torch.tensor([[100.0], [200.0]])
    covariates_name = ["cov1"] if with_covariates else None
    covariates_data = torch.tensor([[8.0], [9.0]]) if with_covariates else None

    out = fedcombat_jobs._save_updated_dataset_values(
        phenotypes_name=["phen1"],
        phenotypes_data=phenotypes_data,
        covariates_name=covariates_name,
        covariates_data=covariates_data,
        is_harmonization=is_harmonization,
    )

    assert out["dataset_id"] == "dyn_ds_1"
    assert "private" not in out
    path = fedcombat_jobs._dataset_manager.last_add_kwargs["path"]
    assert path.endswith(".csv")
    if is_harmonization:
        assert "_harmonized_" in Path(path).name
    else:
        assert "_updated_" in Path(path).name


def test_save_and_read_state_values(monkeypatch, fedcombat_jobs):
    monkeypatch.setattr(module, "Serializer", DummySerializer)
    fedcombat_jobs._harmonization_step = HarmonizationStep.STANDARDIZE

    values = {"x": torch.tensor([1.0])}
    fedcombat_jobs._save_state_values(values)

    assert len(fedcombat_jobs._node_state_manager.add_calls) == 1
    loaded = fedcombat_jobs._read_state_values()
    assert torch.equal(loaded["x"], values["x"])
