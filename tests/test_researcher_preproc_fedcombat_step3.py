"""Tests for _FedCombat_Step3 federated preprocessing workflow - model training."""

import pytest
import torch

import fedbiomed.researcher.federated_workflows as fw
from fedbiomed.common.datamanager import DataManager
from fedbiomed.common.dataset import TabularDataset
from fedbiomed.common.training_args import TrainingArgs
from fedbiomed.researcher.datasets import FederatedDataset
from fedbiomed.researcher.federated_workflows.preproc._fedcombat_step3 import (
    _FedCombat_Step3,
    _FedCombatTrainingPlan,
)


@pytest.fixture
def exp_folder():
    return "test_experiment"


@pytest.fixture
def mock_fds(mocker):
    fds = mocker.MagicMock(spec=FederatedDataset)
    fds.data.return_value = {"n1": {"dataset_id": "ds1"}, "n2": {"dataset_id": "ds2"}}
    fds.node_ids.return_value = ["n1", "n2"]
    return fds


@pytest.fixture
def base_preproc_step3(mock_fds, exp_folder):
    return _FedCombat_Step3(
        fds=mock_fds,
        nodes=["n1", "n2"],
        experimentation_folder=exp_folder,
        covariates=["cov1", "cov2"],
        phenotypes=["phen1", "phen2"],
        training_args={"arg1": "value1", "arg2": "value2"},
        model_args={"argA": "valueA", "argB": "valueB"},
        rounds=2,
    )


@pytest.fixture
def base_preproc_step3_factory(mock_fds, exp_folder):
    def _factory(**preproc_args):
        return _FedCombat_Step3(
            fds=mock_fds,
            nodes=preproc_args.get("nodes", mock_fds.node_ids()),
            experimentation_folder=exp_folder,
            covariates=preproc_args.get("cov", ["cov1", "cov2"]),
            phenotypes=preproc_args.get("phenotypes", ["phen1", "phen2"]),
            training_args=preproc_args.get("training_args", {}),
            model_args=preproc_args.get("model_args", {}),
            rounds=preproc_args.get("rounds", 1),
        )

    return _factory


def test_fedcombat_step3_init_ok(base_preproc_step3, exp_folder):
    """Test successful initialization of _FedCombat_Step3."""

    assert base_preproc_step3._fds is not None
    assert base_preproc_step3._nodes == ["n1", "n2"]
    assert base_preproc_step3._experimentation_folder == exp_folder + "_fedcombat"
    assert base_preproc_step3._covariates == ["cov1", "cov2"]
    assert base_preproc_step3._phenotypes == ["phen1", "phen2"]
    assert base_preproc_step3._training_args["arg1"] == "value1"
    assert base_preproc_step3._training_args["arg2"] == "value2"
    assert base_preproc_step3._model_args["argA"] == "valueA"
    assert base_preproc_step3._model_args["argB"] == "valueB"
    assert base_preproc_step3._rounds == 2


def test_fedcombat_step3_init_ok_alternative(base_preproc_step3_factory):
    """Test successful initialization of _FedCombat_Step3 with alternative arguments."""

    # TrainingArgs object for training_args
    preproc = base_preproc_step3_factory(
        cov=[1, 2],
        phenotypes=[3, 4],
        training_args=TrainingArgs({"epochs": 7}),
        model_args={"argY": "valueY"},
        rounds=5,
    )

    assert preproc._training_args["epochs"] == 7

    # None for training_args
    preproc = base_preproc_step3_factory(
        cov=["covA"],
        phenotypes=["phenB"],
        training_args=None,
        model_args={},
        rounds=3,
    )

    assert preproc._training_args["epochs"] == 1  # default value


def test_fedcombat_step3_init_bad_arguments(base_preproc_step3_factory):
    """Test initialization of _FedCombat_Step3 with bad arguments."""

    # Covariates not a list
    with pytest.raises(Exception) as excinfo:
        base_preproc_step3_factory(cov="not_a_list")
    assert "Covariates and phenotypes must be provided as lists" in str(excinfo.value)

    # Phenotypes not a list
    with pytest.raises(Exception) as excinfo:
        base_preproc_step3_factory(phenotypes="not_a_list")
    assert "Covariates and phenotypes must be provided as lists" in str(excinfo.value)

    # No covariates provided
    with pytest.raises(Exception) as excinfo:
        base_preproc_step3_factory(cov=[])
    assert "At least one covariate must be provided" in str(excinfo.value)

    # No phenotypes provided
    with pytest.raises(Exception) as excinfo:
        base_preproc_step3_factory(phenotypes=[])
    assert "At least one phenotype must be provided" in str(excinfo.value)

    # Mix int and str phenotypes
    with pytest.raises(Exception) as excinfo:
        base_preproc_step3_factory(phenotypes=["phen1", 2])
    assert "Covariates and phenotypes must all be of the same type" in str(
        excinfo.value
    )

    # Mix int and str covariates
    with pytest.raises(Exception) as excinfo:
        base_preproc_step3_factory(cov=["cov1", 2])
    assert "Covariates and phenotypes must all be of the same type" in str(
        excinfo.value
    )

    # Mix int covariates and str phenotypes
    with pytest.raises(Exception) as excinfo:
        base_preproc_step3_factory(cov=[1, 2], phenotypes=["phen1", "phen2"])
    assert "Covariates and phenotypes must all be of the same type" in str(
        excinfo.value
    )

    # Duplicates between covariates and phenotypes
    with pytest.raises(Exception) as excinfo:
        base_preproc_step3_factory(cov=["var1", "var2"], phenotypes=["var2", "var3"])
    assert "Covariates and phenotypes must be disjoint lists of unique values" in str(
        excinfo.value
    )


def test_fedcombat_step3_execute(base_preproc_step3, mocker, monkeypatch):
    """Test execution of _FedCombat_Step3 with mocked training plan."""

    # prepare mock Experiment class returning an instance whose .run() we can assert
    MockExperiment = mocker.MagicMock()
    mock_instance = mocker.MagicMock()
    MockExperiment.return_value = mock_instance

    # patch package attribute so local import inside execute() uses it
    monkeypatch.setattr(fw, "Experiment", MockExperiment, raising=False)

    # 1. execute successfully

    # Execute the preprocessing step
    base_preproc_step3.execute()
    # Check that the training plan was initialized with correct arguments
    mock_instance.run.assert_called_once()

    # 2. execute with failure at Experiment()
    mock_instance.run.reset_mock()
    MockExperiment.side_effect = Exception()
    with pytest.raises(Exception):
        base_preproc_step3.execute()

    # 3. execute with failure at .run()
    mock_instance.run.reset_mock()
    MockExperiment.side_effect = None
    mock_instance.run.side_effect = Exception()
    with pytest.raises(Exception):
        base_preproc_step3.execute()


def test_fedcombat_step3_instantiate_training_plan(mocker):
    # Patch dependencies of _FedCombatTrainingPlan
    mock_loss = mocker.patch(
        "fedbiomed.researcher.federated_workflows.preproc._fedcombat_step3.nn.MSELoss",
        new=mocker.MagicMock(),
    )
    mock_loss.return_value = lambda x, y: float(torch.mean((x - y) ** 2))
    mocker.patch(
        "fedbiomed.researcher.federated_workflows.preproc._fedcombat_step3.TabularDataset",
        mocker.MagicMock(spec=TabularDataset),
    )
    mock_datamanager = mocker.MagicMock(spec=DataManager)
    mock_datamanager.return_value = "training_data_return_value"
    mocker.patch(
        "fedbiomed.researcher.federated_workflows.preproc._fedcombat_step3.DataManager",
        mock_datamanager,
    )
    mocker.patch(
        "fedbiomed.researcher.federated_workflows.preproc._fedcombat_step3.TorchTrainingPlan",
        new=mocker.MagicMock(),
    )

    # Instantiate the training plan
    tp = _FedCombatTrainingPlan()
    model_args = {
        "covariates": ["cov1", "cov2"],
        "phenotypes": ["phen1"],
    }
    tp.model = mocker.MagicMock()
    test_data = torch.randn(2, 2)
    test_target = torch.randn(2, 1)
    tp.model_args = mocker.MagicMock()
    tp.model_args.return_value = model_args

    # Test the methods

    # 1. init_model
    model = tp.init_model(model_args)
    assert isinstance(model, torch.nn.Module)

    tp.model.return_value = model

    # 2. init_optimizer
    optimizer = tp.init_optimizer({"lr": 0.1})
    assert isinstance(optimizer, torch.optim.Optimizer)

    # 3. training_data
    training_data = tp.training_data()
    assert training_data == mock_datamanager.return_value

    # 4. training_step
    training_step = tp.training_step(test_data, test_target)
    assert isinstance(training_step, float)
