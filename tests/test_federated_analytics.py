import uuid
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from fedbiomed.common.exceptions import FedbiomedExperimentError
from fedbiomed.researcher.datasets import FederatedDataset
from fedbiomed.researcher.federated_workflows import FederatedAnalytics
from fedbiomed.researcher.requests import Requests


@pytest.fixture
def mock_fds():
    fds = MagicMock(spec=FederatedDataset)
    fds.node_ids.return_value = ["node-1", "node-2"]
    return fds


@pytest.fixture
def empty_fds():
    fds = MagicMock(spec=FederatedDataset)
    fds.node_ids.return_value = []
    return fds


@pytest.fixture
def mock_requests():
    return MagicMock(spec=Requests)


@pytest.fixture
def base_fa(mock_fds, mock_requests):
    return FederatedAnalytics(
        fds=mock_fds,
        experiment_id="exp-123",
        researcher_id="res-456",
        reqs=mock_requests,
        experimentation_folder="/tmp/fedbiomed",
    )


def test_fa_id_is_created_and_unique(base_fa):
    assert base_fa.fa_id.startswith("FA_")

    # Ensure UUID part is valid
    uuid_part = base_fa.fa_id.replace("FA_", "")
    uuid.UUID(uuid_part)  # will raise if invalid


def test_get_node_ids_success(base_fa, mock_fds):
    node_ids = base_fa.get_node_ids()

    assert node_ids == ["node-1", "node-2"]
    mock_fds.node_ids.assert_called_once()


def test_get_node_ids_no_fds_raises():
    fa = FederatedAnalytics(
        fds=None,
        experiment_id="exp",
        researcher_id="res",
        reqs=MagicMock(),
        experimentation_folder="/tmp",
    )

    with pytest.raises(FedbiomedExperimentError):
        fa.get_node_ids()


def test_get_node_ids_empty_raises(empty_fds, mock_requests):
    fa = FederatedAnalytics(
        fds=empty_fds,
        experiment_id="exp",
        researcher_id="res",
        reqs=mock_requests,
        experimentation_folder="/tmp",
    )

    with pytest.raises(FedbiomedExperimentError):
        fa.get_node_ids()


@patch("fedbiomed.researcher.federated_workflows._federated_analytics.FARequestJob")
def test_mean_executes_fa_job(
    mock_fa_job_cls,
    base_fa,
):
    mock_execute = mock_fa_job_cls.return_value.execute
    fake_result = ({"dummy": "kwargs"}, np.ones((3, 3)))
    mock_execute.return_value = fake_result

    result = base_fa.mean(col_names=["a", "b"])

    # Result is whatever FAResearcherJob.execute returns
    assert result == fake_result

    # Ensure job execution was triggered
    mock_execute.assert_called_once()


def test_mean_without_fds_raises(mock_requests):
    fa = FederatedAnalytics(
        fds=None,
        experiment_id="exp",
        researcher_id="res",
        reqs=mock_requests,
        experimentation_folder="/tmp",
    )

    with pytest.raises(FedbiomedExperimentError):
        fa.mean(col_names=["a"])
