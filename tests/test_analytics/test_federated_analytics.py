import uuid
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from fedbiomed.common.constants import AnalyticsTypes, Stats
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedExperimentError
from fedbiomed.common.message import FAReply
from fedbiomed.researcher.datasets import FederatedDataset
from fedbiomed.researcher.federated_workflows import FederatedAnalytics
from fedbiomed.researcher.federated_workflows._federated_analytics import FAResult
from fedbiomed.researcher.requests import Requests


@pytest.fixture
def mock_fds():
    fds = MagicMock(spec=FederatedDataset)
    fds.data.return_value = {
        "node-1": {"dataset_id": "ds-1", "data_type": "csv"},
        "node-2": {"dataset_id": "ds-2", "data_type": "csv"},
    }
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


class TestFAResult:
    def test_init_raises_if_no_replies(self):
        with pytest.raises(FedbiomedError, match="No replies provided"):
            FAResult({})

    def test_init_raises_if_invalid_aggregator(self):
        reply = MagicMock(spec=FAReply)
        reply.output = {"mod1": {"stat1": 1}}

        # Make sure output validation passes
        with pytest.raises(
            FedbiomedError, match="Aggregators should be provided as a dictionary"
        ):
            FAResult({"n1": reply}, aggregators="not-a-dict")

        with pytest.raises(
            FedbiomedError, match="Aggregator for 'stat1' is not callable"
        ):
            FAResult({"n1": reply}, aggregators={"stat1": "not-callable"})

    def test_values_property(self):
        reply1 = MagicMock(spec=FAReply)
        reply1.output = {"mod1": {"mean": 10}}
        reply2 = MagicMock(spec=FAReply)
        reply2.output = {"mod1": {"mean": 20}}

        res = FAResult({"n1": reply1, "n2": reply2})
        values = res.values

        # Expect {mod1: {mean: [10, 20]}} ignoring order
        assert "mod1" in values
        assert "mean" in values["mod1"]
        # Convert to set for order-independent comparison if needed, or sort
        assert sorted(values["mod1"]["mean"]) == [10, 20]

    def test_aggregate_success(self):
        reply1 = MagicMock(spec=FAReply)
        reply1.output = {"mod1": {"mean": 10}}
        reply2 = MagicMock(spec=FAReply)
        reply2.output = {"mod1": {"mean": 20}}

        res = FAResult(
            {"n1": reply1, "n2": reply2},
            aggregators={"mean": lambda mean: sum(mean) / len(mean)},
        )
        agg = res.aggregate()

        assert agg["mod1"]["mean"] == 15.0

    def test_aggregate_no_aggregators_returns_values(self):
        reply1 = MagicMock(spec=FAReply)
        reply1.output = {"mod1": {"mean": 10}}

        res = FAResult({"n1": reply1})
        assert res.aggregate() == res.values

    def test_inconsistent_modalities_raises(self):
        reply1 = MagicMock(spec=FAReply)
        reply1.output = {"mod1": {"mean": 1}}
        reply2 = MagicMock(spec=FAReply)
        reply2.output = {"mod2": {"mean": 1}}

        with pytest.raises(
            FedbiomedError, match="Nodes present inconsistent modalities"
        ):
            FAResult({"n1": reply1, "n2": reply2})

    def test_inconsistent_statistics_raises(self):
        reply1 = MagicMock(spec=FAReply)
        reply1.output = {"mod1": {"mean": 1}}
        reply2 = MagicMock(spec=FAReply)
        reply2.output = {"mod1": {"max": 1}}

        with pytest.raises(FedbiomedError, match="inconsistent statistics"):
            FAResult({"n1": reply1, "n2": reply2})


class TestFederatedAnalytics:
    def test_fa_id_is_created_and_unique(self, base_fa):
        assert base_fa.fa_id.startswith("FA_")
        uuid_part = base_fa.fa_id.replace("FA_", "")
        uuid.UUID(uuid_part)  # will raise if invalid

    def test_get_node_ids_success(self, base_fa, mock_fds):
        node_ids = base_fa.get_node_ids()
        assert node_ids == ["node-1", "node-2"]
        mock_fds.node_ids.assert_called_once()

    def test_get_node_ids_empty(self, empty_fds, mock_requests):
        fa = FederatedAnalytics(
            fds=empty_fds,
            experiment_id="exp",
            researcher_id="res",
            reqs=mock_requests,
            experimentation_folder="/tmp",
        )
        with pytest.raises(FedbiomedExperimentError):
            fa.get_node_ids()

    def test_get_node_ids_no_fds_raises(self):
        fa = FederatedAnalytics(
            fds=None,
            experiment_id="exp",
            researcher_id="res",
            reqs=MagicMock(),
            experimentation_folder="/tmp",
        )

        with pytest.raises(FedbiomedExperimentError):
            fa.get_node_ids()

    def test_update_fa_args(self, base_fa):
        fa_args = {"existing": 1}
        new_args = base_fa._update_fa_args(fa_args, {"existing": 2, "new": 3})
        assert new_args == {"existing": 2, "new": 3}

        new_args_none = base_fa._update_fa_args(None, {"a": 1})
        assert new_args_none == {"a": 1}

    @patch("fedbiomed.researcher.federated_workflows._federated_analytics.FARequestJob")
    @patch(
        "fedbiomed.researcher.federated_workflows._federated_analytics.validate_dataset_arguments_for_fa"
    )
    # We patch the instance method on the class to avoid needing an instance beforehand
    @patch.object(FederatedAnalytics, "_validate_if_dataset_has_analytics")
    def test_compute_analytics_flow(
        self, mock_validate_analytics, mock_validate_args, mock_fa_job_cls, base_fa
    ):
        replies = {"n1": "r1"}
        errors = {}
        # FARequestJob().execute() returns (replies, errors)
        mock_fa_job_cls.return_value.execute.return_value = (replies, errors)

        res = base_fa._compute_analytics("my_analytics", dataset_args={}, fa_args={})

        assert res == replies
        mock_validate_analytics.assert_called_once_with("my_analytics")
        mock_validate_args.assert_called_once()
        mock_fa_job_cls.assert_called_once()

    @patch("fedbiomed.researcher.federated_workflows._federated_analytics.FARequestJob")
    def test_mean_executes_fa_job(self, mock_fa_job_cls, base_fa):
        """This replaces the old test_mean_executes_fa_job with correct assertions."""
        # Setup mocks
        replies = {"node-1": MagicMock(spec=FAReply, output={"mod1": {"mean": 5.0}})}
        errors = {}
        mock_fa_job_cls.return_value.execute.return_value = (replies, errors)

        # We need to ensure validations pass
        with (
            patch.object(base_fa, "_validate_if_dataset_has_analytics"),
            patch.object(base_fa, "_validate_dataset_arguments"),
        ):
            result = base_fa.mean(dataset_args={"col_names": ["mod1"]})

            assert isinstance(result, FAResult)
            result_vals = result.values
            assert result_vals["mod1"]["mean"] == [5.0]

    @patch.object(FederatedAnalytics, "_compute_analytics")
    def test_basic_stats_wrapper(self, mock_compute, base_fa):
        reply_obj = MagicMock(
            spec=FAReply,
            output={"mod": {"mean": 1, "min": 0, "max": 2, "std": 0.5, "count": 10}},
        )
        mock_compute.return_value = {"n1": reply_obj}

        result = base_fa.basic_stats(dataset_args={})

        assert isinstance(result, FAResult)
        mock_compute.assert_called_once_with(AnalyticsTypes.BASIC_STATS.value, {}, None)
        assert "mean" in result.values["mod"]

    @patch.object(FederatedAnalytics, "_compute_analytics")
    def test_mean_wrapper(self, mock_compute, base_fa):
        reply_obj = MagicMock(spec=FAReply, output={"mod": {"mean": 1}})
        mock_compute.return_value = {"n1": reply_obj}

        result = base_fa.mean(dataset_args={})

        assert isinstance(result, FAResult)
        call_args = mock_compute.call_args
        assert call_args[0][0] == AnalyticsTypes.BASIC_STATS.value
        assert call_args[0][2]["requested_stats"] == [Stats.MEAN.value]

    @patch.object(FederatedAnalytics, "_compute_analytics")
    def test_min_max_wrapper(self, mock_compute, base_fa):
        reply_obj = MagicMock(spec=FAReply, output={"mod": {"min": 0, "max": 1}})
        mock_compute.return_value = {"n1": reply_obj}

        base_fa.min_max()
        call_args = mock_compute.call_args
        assert call_args[0][2]["requested_stats"] == [Stats.MIN.value, Stats.MAX.value]

    @patch.object(FederatedAnalytics, "_compute_analytics")
    def test_sum_wrapper(self, mock_compute, base_fa):
        reply_obj = MagicMock(spec=FAReply, output={"mod": {"sum": 10}})
        mock_compute.return_value = {"n1": reply_obj}

        base_fa.sum()
        call_args = mock_compute.call_args
        assert call_args[0][2]["requested_stats"] == [Stats.SUM.value]

    @patch.object(FederatedAnalytics, "_compute_analytics")
    def test_histogram_wrapper(self, mock_compute, base_fa):
        # We need to mock _create_bins
        with patch.object(
            base_fa, "_create_bins", return_value={"mod": [0, 1]}
        ) as mock_create_bins:
            reply_obj = MagicMock(
                spec=FAReply, output={"mod": {"counts": [5], "bin_edges": [0, 1]}}
            )
            mock_compute.return_value = {"n1": reply_obj}

            base_fa.histogram(num_bins=5)

            mock_create_bins.assert_called_once()
            call_args = mock_compute.call_args
            assert call_args[0][0] == AnalyticsTypes.HISTOGRAM.value
            assert call_args[0][2]["bin_edges"] == {"mod": [0, 1]}

    @patch.object(FederatedAnalytics, "_compute_analytics")
    def test_create_bins(self, mock_compute, base_fa):
        # Result of min_max analytics passed to min_max wrapper
        reply1 = MagicMock(spec=FAReply, output={"mod1": {"min": 0.0, "max": 10.0}})
        reply2 = MagicMock(spec=FAReply, output={"mod1": {"min": 2.0, "max": 12.0}})
        mock_compute.return_value = {"n1": reply1, "n2": reply2}

        bin_edges = base_fa._create_bins(num_bins=10)

        # Global min 0.0, global max 12.0
        # Margin = 12 * 0.001 = 0.012
        # Edges from 0 to 12.012, 11 edges (10 bins)
        assert "mod1" in bin_edges
        assert len(bin_edges["mod1"]) == 11
        assert np.isclose(bin_edges["mod1"][0], 0.0)
        assert bin_edges["mod1"][-1] > 12.0

    # TODO: Fix this test
    """
    @patch.object(FederatedAnalytics, "_compute_analytics")
    def test_quantile_wrapper(self, mock_compute, base_fa):
        # Mock _create_bins
        with patch.object(base_fa, "_create_bins", return_value={"mod": [0, 1]}):
            # Mock return of _compute_analytics to be a tuple (node_quantiles, errors)
            # This is validating the expectation of CURRENT code, even if flawed.

            node_quantiles = {
                "n1": MagicMock(spec=FAReply, output={"mod1": {"q0.5": 10.0}})
            }
            errors = {}
            mock_compute.return_value = (node_quantiles, errors)

            agg_q, node_q, err = base_fa.quantile()

            assert agg_q["mod1"]["q0.5"] == 10.0
            assert node_q == node_quantiles
            assert err == errors
    """
