import uuid
from unittest.mock import MagicMock, patch

import pytest

from fedbiomed.common.constants import SAParameters
from fedbiomed.common.exceptions import (
    FedbiomedError,
    FedbiomedExperimentError,
    FedbiomedSecureAggregationError,
)
from fedbiomed.common.message import FAReply
from fedbiomed.researcher.datasets import FederatedDataset
from fedbiomed.researcher.federated_workflows import FederatedAnalytics
from fedbiomed.researcher.federated_workflows._federated_analytics import FAResult
from fedbiomed.researcher.requests import Requests

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


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


def _make_reply(output: dict) -> MagicMock:
    """Create a MagicMock FAReply with the given output dict (plaintext path)."""
    r = MagicMock(spec=FAReply)
    r.output = output
    r.encrypted = False
    return r


# ---------------------------------------------------------------------------
# TestFAResult
# ---------------------------------------------------------------------------
#
# Test fixtures use realistic schema-based output shapes:
#
#   ROW (bare RowSpec):    {col: {stat: val}}
#   IMAGE (flat):          {stat: val}            (ImageSpec)
#   Nested dict schema:    {key: {col: {stat: val}}}
#   Sequence schema:       ({col: {stat: val}}, {stat: val})
#


class TestFAResult:
    # --- construction & basic properties ---

    def test_empty_init(self):
        result = FAResult({})
        assert result.node_ids == []
        assert result.available_stats() == []

    def test_single_node_row_schema(self):
        # ROW output: {col: {stat: val}}
        replies = {"n1": _make_reply({"age": {"mean": 45.0, "count": 100}})}
        result = FAResult(replies)
        assert result.node_ids == ["n1"]

    def test_multiple_nodes(self):
        replies = {
            "n1": _make_reply({"age": {"mean": 45.0, "count": 100}}),
            "n2": _make_reply({"age": {"mean": 50.0, "count": 80}}),
        }
        result = FAResult(replies)
        assert sorted(result.node_ids) == ["n1", "n2"]

    def test_none_output_raises(self):
        with pytest.raises(FedbiomedError):
            FAResult({"n1": _make_reply(None)})

    # --- available_stats ---

    def test_available_stats_row(self):
        replies = {"n1": _make_reply({"age": {"mean": 45.0, "count": 100}})}
        result = FAResult(replies)
        assert sorted(result.available_stats()) == ["count", "mean"]

    def test_available_stats_image_flat(self):
        # IMAGE (flat): top-level dict is itself a stat-leaf — no column-key wrapper
        replies = {"n1": _make_reply({"mean": 128.0, "count": 100})}
        result = FAResult(replies)
        assert sorted(result.available_stats()) == ["count", "mean"]

    def test_available_stats_empty(self):
        assert FAResult({}).available_stats() == []

    # --- schema ---

    def test_schema_empty(self):
        assert FAResult({}).schema is None

    def test_schema_row_single_column(self):
        replies = {"n1": _make_reply({"age": {"mean": 45.0, "count": 100}})}
        result = FAResult(replies)
        assert result.schema == {"age": {}}

    def test_schema_row_multiple_columns(self):
        replies = {
            "n1": _make_reply(
                {
                    "age": {"mean": 45.0, "count": 100},
                    "height": {"mean": 170.0, "count": 100},
                }
            )
        }
        result = FAResult(replies)
        assert result.schema == {"age": {}, "height": {}}

    def test_schema_image_flat(self):
        """IMAGE (flat) — top-level dict is itself a stat leaf → {}."""
        replies = {"n1": _make_reply({"mean": 128.0, "count": 100})}
        result = FAResult(replies)
        assert result.schema == {}

    def test_schema_nested_dict_schema(self):
        replies = {
            "n1": _make_reply(
                {
                    "group_a": {"col1": {"mean": 1.0}},
                    "group_b": {"col2": {"mean": 2.0}},
                }
            )
        }
        result = FAResult(replies)
        assert result.schema == {"group_a": {"col1": {}}, "group_b": {"col2": {}}}

    def test_schema_sequence_tuple(self):
        replies = {"n1": _make_reply(({"col1": {"mean": 1.0}}, {"mean": 128.0}))}
        result = FAResult(replies)
        assert result.schema == ({"col1": {}}, {})

    def test_schema_sequence_list(self):
        replies = {"n1": _make_reply([{"col1": {"mean": 1.0}}, {"mean": 128.0}])}
        result = FAResult(replies)
        assert result.schema == [{"col1": {}}, {}]

    def test_schema_preserves_sequence_type(self):
        tuple_replies = {"n1": _make_reply(({"mean": 1.0}, {"count": 2}))}
        list_replies = {"n1": _make_reply([{"mean": 1.0}, {"count": 2}])}
        assert isinstance(FAResult(tuple_replies).schema, tuple)
        assert isinstance(FAResult(list_replies).schema, list)

    def test_schema_uses_first_node_only(self):
        replies = {
            "n1": _make_reply({"age": {"mean": 45.0, "count": 100}}),
            "n2": _make_reply(
                {"age": {"mean": 50.0, "count": 80}, "height": {"mean": 170.0}}
            ),
        }
        result = FAResult(replies)
        assert result.schema == {"age": {}}

    def test_schema_after_merge(self):
        result = FAResult({"n1": _make_reply({"age": {"mean": 45.0, "count": 100}})})
        result.merge(
            {"n1": _make_reply({"age": {"variance": 4.0, "mean": 45.0, "count": 100}})}
        )
        assert result.schema == {"age": {}}

    def test_schema_scalar_non_stat_leaf(self):
        # A dict whose values are raw scalars (not stat-leaf dicts) — _schema returns
        # None for each scalar child (covers line 179: `return None`)
        replies = {"n1": _make_reply({"label": "sample", "version": 2})}
        result = FAResult(replies)
        assert result.schema == {"label": None, "version": None}

    def test_first_output_raises_on_empty_data(self):
        # _first_output guard raises when _data is empty (covers line 83)
        result = FAResult({})
        with pytest.raises(FedbiomedError, match="contains no node data"):
            _ = result._first_output

    # --- node_stats ---

    def test_node_stats_row(self):
        replies = {"n1": _make_reply({"age": {"mean": 45.0, "count": 100}})}
        result = FAResult(replies)
        assert result.node_stats("n1") == {"age": {"mean": 45.0, "count": 100}}

    def test_node_stats_image_flat(self):
        replies = {"n1": _make_reply({"mean": 128.0, "count": 100})}
        result = FAResult(replies)
        assert result.node_stats("n1") == {"mean": 128.0, "count": 100}

    def test_node_stats_invalid_node_raises(self):
        replies = {"n1": _make_reply({"age": {"mean": 45.0}})}
        result = FAResult(replies)
        with pytest.raises(FedbiomedError, match="not found"):
            result.node_stats("unknown-node")

    # --- global_stats ---

    def test_global_stats_mean_row(self):
        # ROW: {col: {stat: val}} — result is {col: aggregated_value}
        # Nodes return summable primitives: sum + count
        replies = {
            "n1": _make_reply({"age": {"sum": 4500.0, "count": 100}}),
            "n2": _make_reply({"age": {"sum": 4000.0, "count": 80}}),
        }
        result = FAResult(replies)
        global_mean = result.global_stats("mean")
        expected_age = (45.0 * 100 + 50.0 * 80) / 180
        assert isinstance(global_mean, dict)
        assert abs(global_mean["age"] - expected_age) < 1e-9

    def test_global_stats_mean_image_flat(self):
        # IMAGE: {stat: val} — result is a scalar directly
        # Nodes return summable primitives: sum + count
        replies = {
            "n1": _make_reply({"sum": 12800.0, "count": 100}),
            "n2": _make_reply({"sum": 10400.0, "count": 80}),
        }
        result = FAResult(replies)
        global_mean = result.global_stats("mean")
        expected = (128.0 * 100 + 130.0 * 80) / 180
        assert abs(global_mean - expected) < 1e-9

    def test_global_stats_sum_row(self):
        # Nodes return summable primitives: sum + count
        replies = {
            "n1": _make_reply({"age": {"sum": 4500.0, "count": 100}}),
            "n2": _make_reply({"age": {"sum": 4000.0, "count": 80}}),
        }
        result = FAResult(replies)
        global_sum = result.global_stats("sum")
        expected_sum = 4500.0 + 4000.0
        assert isinstance(global_sum, dict)
        assert abs(global_sum["age"] - expected_sum) < 1e-9

    def test_global_stats_variance_row(self):
        # Two-pass centered primitives: count + Σ(x − μ_global)².
        # Global ages [40,50,60], μ=50, N=3 → var(ddof=1) = (100+0+100)/2 = 100.
        # node1 = [40,50] → Σ(x−50)² = 100 ; node2 = [60] → Σ(x−50)² = 100.
        replies = {
            "n1": _make_reply({"age": {"sum_sq_centered": 100.0, "count": 2}}),
            "n2": _make_reply({"age": {"sum_sq_centered": 100.0, "count": 1}}),
        }
        result = FAResult(replies)
        global_variance = result.global_stats("variance")
        assert isinstance(global_variance, dict)
        assert global_variance["age"] == pytest.approx(100.0)

    def test_global_stats_count_row(self):
        replies = {
            "n1": _make_reply({"age": {"count": 100}}),
            "n2": _make_reply({"age": {"count": 80}}),
        }
        result = FAResult(replies)
        assert result.global_stats("count") == {"age": 180}

    def test_global_stats_nested_dict_schema(self):
        # Nested: {key: {col: {stat: val}}} — result is {key: {col: val}}
        # Nodes return summable primitives: sum + count
        replies = {
            "n1": _make_reply({"tabular": {"age": {"sum": 4500.0, "count": 100}}}),
            "n2": _make_reply({"tabular": {"age": {"sum": 4000.0, "count": 80}}}),
        }
        result = FAResult(replies)
        global_mean = result.global_stats("mean")
        expected_age = (45.0 * 100 + 50.0 * 80) / 180
        assert isinstance(global_mean, dict)
        assert abs(global_mean["tabular"]["age"] - expected_age) < 1e-9

    def test_global_stats_missing_data_raises(self):
        # variance is valid but requires data not present → "cannot be computed"
        replies = {"n1": _make_reply({"age": {"mean": 45.0, "count": 100}})}
        result = FAResult(replies)
        with pytest.raises(FedbiomedError, match="cannot be computed"):
            result.global_stats("variance")

    def test_global_stats_unknown_stat_raises(self):
        # "skewness" is not a registered statistic at all → "not a valid statistic"
        replies = {"n1": _make_reply({"age": {"mean": 45.0, "count": 100}})}
        result = FAResult(replies)
        with pytest.raises(FedbiomedError, match="not a valid statistic"):
            result.global_stats("skewness")

    # --- merge ---

    def test_merge_adds_new_stat(self):
        result = FAResult({"n1": _make_reply({"age": {"mean": 45.0, "count": 100}})})
        new_replies = {
            "n1": _make_reply({"age": {"variance": 4.0, "mean": 45.0, "count": 100}})
        }
        result.merge(new_replies)
        assert "variance" in result.available_stats()
        assert "mean" in result.available_stats()

    def test_merge_preserves_existing_stats(self):
        result = FAResult({"n1": _make_reply({"age": {"mean": 45.0, "count": 100}})})
        new_replies = {
            "n1": _make_reply({"age": {"mean": 45.0, "count": 100, "variance": 4.0}})
        }
        result.merge(new_replies)
        assert "mean" in result.available_stats()
        assert "variance" in result.available_stats()

    def test_merge_adds_new_node(self):
        result = FAResult({"n1": _make_reply({"age": {"mean": 45.0, "count": 100}})})
        new_replies = {"n2": _make_reply({"age": {"mean": 50.0, "count": 80}})}
        result.merge(new_replies)
        assert "n2" in result.node_ids

    def test_merge_invalid_output_raises(self):
        result = FAResult({"n1": _make_reply({"age": {"mean": 45.0}})})
        with pytest.raises(FedbiomedError):
            result.merge({"n2": _make_reply(None)})

    def test_merge_deep_merges_nested_structure(self):
        # Start with mean; merge in variance at the same col
        result = FAResult({"n1": _make_reply({"age": {"mean": 45.0, "count": 100}})})
        result.merge(
            {"n1": _make_reply({"age": {"variance": 4.0, "mean": 45.0, "count": 100}})}
        )
        stats = result.node_stats("n1")
        assert "variance" in stats["age"]
        assert "mean" in stats["age"]

    def test_deep_merge_mismatched_sequences_raises(self):
        with pytest.raises(FedbiomedError):
            FAResult._deep_merge((1, 2, 3), (1, 2))

    def test_deep_merge_sequences_success(self):
        # Same-type, same-length sequences are merged element-wise (covers line 111)
        result = FAResult._deep_merge(
            [{"age": {"mean": 1.0}}, {"age": {"mean": 2.0}}],
            [{"age": {"mean": 3.0}}, {"age": {"mean": 4.0}}],
        )
        assert result == [{"age": {"mean": 3.0}}, {"age": {"mean": 4.0}}]
        assert isinstance(result, list)

        result_tuple = FAResult._deep_merge(
            ({"age": {"mean": 1.0}},),
            ({"age": {"mean": 5.0}},),
        )
        assert isinstance(result_tuple, tuple)
        assert result_tuple == ({"age": {"mean": 5.0}},)

    def test_deep_merge_incompatible_types_raises(self):
        with pytest.raises(FedbiomedError):
            FAResult._deep_merge({"a": 1}, [1])

        with pytest.raises(FedbiomedError):
            FAResult._deep_merge([1], {"a": 1})

        with pytest.raises(FedbiomedError):
            # One sequence, one scalar
            FAResult._deep_merge([1], 1)

    # --- node_stats ---

    def test_node_stats_returns_dict_by_default(self):
        replies = {
            "n1": _make_reply({"age": {"mean": 45.0, "count": 100}}),
            "n2": _make_reply({"age": {"mean": 50.0, "count": 80}}),
        }
        result = FAResult(replies)
        all_stats = result.node_stats()
        assert isinstance(all_stats, dict)
        assert set(all_stats.keys()) == {"n1", "n2"}
        assert all_stats["n1"] == result.node_stats("n1")
        assert all_stats["n2"] == result.node_stats("n2")

    def test_node_stats_empty_result(self):
        result = FAResult({})
        assert result.node_stats() == {}


# ---------------------------------------------------------------------------
# TestFederatedAnalytics
# ---------------------------------------------------------------------------


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

    @patch("fedbiomed.researcher.federated_workflows._federated_analytics.FARequestJob")
    def test_fetch_stats_returns_fa_result(self, mock_fa_job_cls, base_fa):
        replies = {"n1": _make_reply({"age": {"mean": 45.0, "count": 100}})}
        mock_fa_job_cls.return_value.execute.return_value = replies

        result = base_fa.fetch_stats("mean")

        assert isinstance(result, FAResult)
        mock_fa_job_cls.assert_called_once()

    @patch("fedbiomed.researcher.federated_workflows._federated_analytics.FARequestJob")
    def test_mean_returns_fa_result(self, mock_fa_job_cls, base_fa):
        replies = {"node-1": _make_reply({"age": {"sum": 4500.0, "count": 100}})}
        mock_fa_job_cls.return_value.execute.return_value = replies

        result = base_fa.mean()

        assert isinstance(result, dict)
        assert "age" in result

    # --- Caching tests ---

    @patch("fedbiomed.researcher.federated_workflows._federated_analytics.FARequestJob")
    def test_same_stat_same_args_uses_cache(self, mock_fa_job_cls, base_fa):
        """Second call for the same primitive stat must be served from cache.

        Primitive stats (count, sum, sum_sq_centered) are stored as-is, so the
        cache-hit check finds them already satisfiable (present as a leaf or
        derivable from cached primitives) and skips the second request.
        """
        replies = {"node-1": _make_reply({"age": {"count": 100}})}
        mock_fa_job_cls.return_value.execute.return_value = replies

        result1 = base_fa.count()
        result2 = base_fa.count()

        assert mock_fa_job_cls.call_count == 1
        assert result1 == result2

    @patch("fedbiomed.researcher.federated_workflows._federated_analytics.FARequestJob")
    def test_fetch_stats_with_args_different_args_bypass_cache(
        self, mock_fa_job_cls, base_fa
    ):
        """Different stats_args must cause a separate network request."""
        replies = {
            "node-1": _make_reply(
                {"age": {"histogram": {"bin_edges": [0, 1], "counts": [5]}}}
            )
        }
        mock_fa_job_cls.return_value.execute.return_value = replies

        base_fa.fetch_stats_with_args(
            stats_args={"age": {"histogram": {"bin_edges": [0, 1]}}}
        )
        base_fa.fetch_stats_with_args(
            stats_args={"age": {"histogram": {"bin_edges": [0, 2]}}}
        )

        assert mock_fa_job_cls.call_count == 2

    @patch("fedbiomed.researcher.federated_workflows._federated_analytics.FARequestJob")
    def test_cached_dependency_avoids_request(self, mock_fa_job_cls, base_fa):
        """
        After requesting mean (which returns sum+count primitives),
        a subsequent count() with the same args must be served from cache
        because 'count' is already present in the cached primitives.
        """
        mean_replies = {"node-1": _make_reply({"age": {"sum": 4500.0, "count": 100}})}
        mock_fa_job_cls.return_value.execute.return_value = mean_replies

        base_fa.mean()
        assert mock_fa_job_cls.call_count == 1

        result = base_fa.count()
        assert mock_fa_job_cls.call_count == 1  # no new network call

        assert isinstance(result, dict)
        assert "age" in result

    @patch("fedbiomed.researcher.federated_workflows._federated_analytics.FARequestJob")
    def test_only_missing_stats_requested(self, mock_fa_job_cls, base_fa):
        """With the mean cached, variance only triggers the centered (round-2) request."""
        mean_replies = {"node-1": _make_reply({"age": {"sum": 4500.0, "count": 100}})}
        mock_fa_job_cls.return_value.execute.return_value = mean_replies
        base_fa.mean()
        assert mock_fa_job_cls.call_count == 1

        # Round 2 reply: count + centered second moment Σ(x − μ)².
        centered_replies = {
            "node-1": _make_reply({"age": {"sum_sq_centered": 396.0, "count": 100}})
        }
        mock_fa_job_cls.return_value.execute.return_value = centered_replies
        result = base_fa.fetch_stats("variance")

        # Round 1 (mean) is served from cache → only the round-2 request hits nodes.
        assert mock_fa_job_cls.call_count == 2
        second_call_kwargs = mock_fa_job_cls.call_args_list[1].kwargs
        # Round 2 is a stats_args-only request (stats / dataset_schema must be None,
        # as FARequestJob forbids combining them with stats_args).
        assert second_call_kwargs["stats"] is None
        assert second_call_kwargs["dataset_schema"] is None
        age_args = second_call_kwargs["stats_args"]["age"]
        assert age_args["sum_sq_centered"]["mean"] == pytest.approx(45.0)

        assert isinstance(result, FAResult)
        assert "variance" in result.computable_stats()

    @patch("fedbiomed.researcher.federated_workflows._federated_analytics.FARequestJob")
    def test_repeated_derived_stat_served_from_cache(self, mock_fa_job_cls, base_fa):
        """A second variance/std request hits the cache, issuing no new node requests."""
        round1 = {"node-1": _make_reply({"age": {"sum": 4500.0, "count": 100}})}
        round2 = {
            "node-1": _make_reply({"age": {"sum_sq_centered": 396.0, "count": 100}})
        }
        mock_fa_job_cls.return_value.execute.side_effect = [round1, round2]

        first = base_fa.fetch_stats("variance")
        assert mock_fa_job_cls.call_count == 2  # round 1 + round 2

        # Repeat the exact same request: must be served entirely from cache.
        second = base_fa.fetch_stats("variance")
        assert mock_fa_job_cls.call_count == 2  # unchanged — no new node requests
        assert second is first  # same cached FAResult object
        assert second.global_stats("variance") == first.global_stats("variance")

    @patch("fedbiomed.researcher.federated_workflows._federated_analytics.FARequestJob")
    def test_repeated_mean_served_from_cache(self, mock_fa_job_cls, base_fa):
        """Repeated mean requests hit the cache (mean is derived from sum+count)."""
        mock_fa_job_cls.return_value.execute.return_value = {
            "node-1": _make_reply({"age": {"sum": 470.0, "count": 10}})
        }

        base_fa.fetch_stats("mean")
        assert mock_fa_job_cls.call_count == 1

        base_fa.mean()  # shorthand -> fetch_stats("mean")
        base_fa.fetch_stats("mean")  # explicit repeat
        base_fa.fetch_stats("count")  # also derivable from the cached primitives
        base_fa.fetch_stats("sum")
        assert mock_fa_job_cls.call_count == 1  # no further node requests

    @patch("fedbiomed.researcher.federated_workflows._federated_analytics.FARequestJob")
    def test_compute_multiple_stats_at_once(self, mock_fa_job_cls, base_fa):
        # Two-pass: round 1 fetches the mean primitives, round 2 the centered moment.
        round1 = {"node-1": _make_reply({"age": {"sum": 4500.0, "count": 100}})}
        round2 = {
            "node-1": _make_reply({"age": {"sum_sq_centered": 396.0, "count": 100}})
        }
        mock_fa_job_cls.return_value.execute.side_effect = [round1, round2]

        result = base_fa.fetch_stats(stats=["mean", "variance"])

        assert "mean" in result.computable_stats()
        assert "variance" in result.computable_stats()
        assert mock_fa_job_cls.call_count == 2

    @patch("fedbiomed.researcher.federated_workflows._federated_analytics.FARequestJob")
    def test_node_change_invalidates_cache(self, mock_fa_job_cls, base_fa, mock_fds):
        """Adding or removing a node creates a new cache entry."""
        replies = {"node-1": _make_reply({"age": {"sum": 4500.0, "count": 100}})}
        mock_fa_job_cls.return_value.execute.return_value = replies

        base_fa.mean()
        assert mock_fa_job_cls.call_count == 1

        # Simulate a node being added to the federation
        mock_fds.node_ids.return_value = ["node-1", "node-2", "node-3"]
        replies_3 = {
            "node-1": _make_reply({"age": {"sum": 4500.0, "count": 100}}),
            "node-3": _make_reply({"age": {"sum": 2880.0, "count": 60}}),
        }
        mock_fa_job_cls.return_value.execute.return_value = replies_3

        base_fa.mean()
        assert mock_fa_job_cls.call_count == 2  # new node set → cache miss

    # --- stats=None (args-only) ---

    @patch("fedbiomed.researcher.federated_workflows._federated_analytics.FARequestJob")
    def test_fetch_stats_with_args_returns_fa_result(self, mock_fa_job_cls, base_fa):
        """fetch_stats_with_args with valid args should issue a request and return FAResult."""
        replies = {"node-1": _make_reply({"image": {"histogram": [0.1, 0.5, 0.9]}})}
        mock_fa_job_cls.return_value.execute.return_value = replies

        result = base_fa.fetch_stats_with_args(
            stats_args={"image": {"histogram": {"bin_edges": [0, 1, 2]}}}
        )

        assert isinstance(result, FAResult)
        mock_fa_job_cls.assert_called_once()
        call_kwargs = mock_fa_job_cls.call_args.kwargs
        assert call_kwargs["stats"] is None

    @patch("fedbiomed.researcher.federated_workflows._federated_analytics.FARequestJob")
    def test_fetch_stats_with_args_cached(self, mock_fa_job_cls, base_fa):
        """Second call with identical stats_args must be served from cache."""
        replies = {"node-1": _make_reply({"image": {"histogram": [0.1, 0.5, 0.9]}})}
        mock_fa_job_cls.return_value.execute.return_value = replies

        args = {"image": {"histogram": {"bin_edges": [0, 1, 2]}}}
        base_fa.fetch_stats_with_args(args)
        base_fa.fetch_stats_with_args(args)

        assert mock_fa_job_cls.call_count == 1  # second call served from cache

    @patch("fedbiomed.researcher.federated_workflows._federated_analytics.FARequestJob")
    def test_fetch_stats_none_defaults_to_count_mean_variance(
        self, mock_fa_job_cls, base_fa
    ):
        """fetch_stats() defaults to count, mean, variance (variance via two-pass)."""
        round1 = {"node-1": _make_reply({"age": {"sum": 4500.0, "count": 100}})}
        round2 = {
            "node-1": _make_reply({"age": {"sum_sq_centered": 396.0, "count": 100}})
        }
        mock_fa_job_cls.return_value.execute.side_effect = [round1, round2]

        result = base_fa.fetch_stats()

        assert isinstance(result, FAResult)
        # Round 1 requests the direct primitives plus count+sum (mean's building
        # blocks); variance is derived from the round-2 centered moment.
        first_call_kwargs = mock_fa_job_cls.call_args_list[0].kwargs
        assert sorted(first_call_kwargs["stats"]) == ["count", "mean", "sum"]
        assert {"count", "mean", "variance"}.issubset(set(result.computable_stats()))

    def test_fetch_stats_empty_list_raises(self, base_fa):
        """Passing an explicit empty list must raise FedbiomedError."""
        with pytest.raises(FedbiomedError, match="stats"):
            base_fa.fetch_stats([])

    def test_fetch_stats_unknown_stat_raises(self, base_fa):
        """Requesting a stat that is not registered at all raises FedbiomedError."""
        with pytest.raises(FedbiomedError, match="not valid"):
            base_fa.fetch_stats("skewness")

    def test_fetch_stats_computed_only_stat_raises(self, base_fa):
        """Requesting a derived stat (e.g. 'quantile') that cannot be requested from nodes raises."""
        with pytest.raises(FedbiomedError, match="derived"):
            base_fa.fetch_stats("quantile")

    def test_fetch_stats_mixed_invalid_stats_raises(self, base_fa):
        """Unknown and derived stats in the same call both appear in the error."""
        with pytest.raises(FedbiomedError) as exc_info:
            base_fa.fetch_stats(["skewness", "quantile"])
        msg = str(exc_info.value)
        assert "skewness" in msg
        assert "quantile" in msg

    def test_fetch_stats_with_args_empty_raises(self, base_fa):
        """Calling fetch_stats_with_args with empty dict must raise FedbiomedError."""
        with pytest.raises(FedbiomedError, match="stats_args"):
            base_fa.fetch_stats_with_args(stats_args={})

    @patch("fedbiomed.researcher.federated_workflows._federated_analytics.FARequestJob")
    def test_fetch_stats_and_fetch_stats_with_args_independent(
        self, mock_fa_job_cls, base_fa
    ):
        """fetch_stats and fetch_stats_with_args use separate cache entries and never share state."""
        stats_replies = {"node-1": _make_reply({"age": {"mean": 45.0, "count": 100}})}
        args_replies = {
            "node-1": _make_reply({"image": {"histogram": [0.1, 0.5, 0.9]}})
        }
        mock_fa_job_cls.return_value.execute.side_effect = [
            stats_replies,
            args_replies,
        ]

        base_fa.fetch_stats("mean")
        base_fa.fetch_stats_with_args(
            {"image": {"histogram": {"bin_edges": [0, 1, 2]}}}
        )

        assert mock_fa_job_cls.call_count == 2

    @patch("fedbiomed.researcher.federated_workflows._federated_analytics.FARequestJob")
    def test_fetch_stats_dataset_schema_affects_cache_key(
        self, mock_fa_job_cls, base_fa
    ):
        """Different dataset_schema values must produce separate cache entries."""
        replies = {"node-1": _make_reply({"age": {"mean": 45.0, "count": 100}})}
        mock_fa_job_cls.return_value.execute.return_value = replies

        base_fa.fetch_stats("mean", dataset_schema=["age"])
        base_fa.fetch_stats("mean", dataset_schema=["height"])

        assert mock_fa_job_cls.call_count == 2

    @patch("fedbiomed.researcher.federated_workflows._federated_analytics.FARequestJob")
    def test_fetch_stats_all_errors_raises(self, mock_fa_job_cls, base_fa):
        """When execute() raises (all nodes failed), FedbiomedError propagates."""
        mock_fa_job_cls.return_value.execute.side_effect = FedbiomedError(
            "all nodes failed"
        )

        with pytest.raises(FedbiomedError):
            base_fa.fetch_stats("mean")

    @patch("fedbiomed.researcher.federated_workflows._federated_analytics.FARequestJob")
    def test_fetch_stats_with_args_all_errors_raises(self, mock_fa_job_cls, base_fa):
        """When execute() raises for fetch_stats_with_args, FedbiomedError propagates."""
        mock_fa_job_cls.return_value.execute.side_effect = FedbiomedError(
            "all nodes failed"
        )

        with pytest.raises(FedbiomedError):
            base_fa.fetch_stats_with_args(
                {"image": {"histogram": {"bin_edges": [0, 1]}}}
            )

    @patch("fedbiomed.researcher.federated_workflows._federated_analytics.FARequestJob")
    def test_partial_errors_raises(self, mock_fa_job_cls, base_fa):
        """When any node returns an error, execute() raises FedbiomedError."""
        mock_fa_job_cls.return_value.execute.side_effect = FedbiomedError(
            "node-2 failed"
        )

        with pytest.raises(FedbiomedError):
            base_fa.fetch_stats("mean")

    @patch("fedbiomed.researcher.federated_workflows._federated_analytics.FARequestJob")
    def test_count_convenience_method(self, mock_fa_job_cls, base_fa):
        """count() delegates to fetch_stats and returns the global count."""
        replies = {
            "node-1": _make_reply({"age": {"count": 100}}),
            "node-2": _make_reply({"age": {"count": 80}}),
        }
        mock_fa_job_cls.return_value.execute.return_value = replies

        result = base_fa.count()

        assert isinstance(result, dict)
        assert result == {"age": 180}

    @patch("fedbiomed.researcher.federated_workflows._federated_analytics.FARequestJob")
    def test_std_convenience_method(self, mock_fa_job_cls, base_fa):
        """std() runs the two-pass centered scheme and returns the global std."""
        # Global ages: node1=[40,50], node2=[60]; μ=50, N=3 → var=100, std=10.
        round1 = {
            "node-1": _make_reply({"age": {"sum": 90.0, "count": 2}}),
            "node-2": _make_reply({"age": {"sum": 60.0, "count": 1}}),
        }
        round2 = {
            "node-1": _make_reply({"age": {"sum_sq_centered": 100.0, "count": 2}}),
            "node-2": _make_reply({"age": {"sum_sq_centered": 100.0, "count": 1}}),
        }
        mock_fa_job_cls.return_value.execute.side_effect = [round1, round2]

        result = base_fa.std()

        assert isinstance(result, dict)
        assert result["age"] == pytest.approx(10.0)

    @patch("fedbiomed.researcher.federated_workflows._federated_analytics.FARequestJob")
    def test_cache_eviction_fifo(self, mock_fa_job_cls, base_fa):
        """Filling cache beyond MAX_CACHE_SIZE evicts the oldest entry (covers line 519)."""
        replies = {"node-1": _make_reply({"age": {"count": 100}})}
        mock_fa_job_cls.return_value.execute.return_value = replies

        max_size = FederatedAnalytics._MAX_CACHE_SIZE
        # Each call uses a distinct stats_args so every result gets its own cache key.
        for i in range(max_size + 1):
            base_fa.fetch_stats_with_args(stats_args={"iteration": i})

        assert mock_fa_job_cls.call_count == max_size + 1
        assert len(base_fa._results_store) == max_size

    @patch("fedbiomed.researcher.federated_workflows._federated_analytics.FARequestJob")
    def test_fetch_stats_string_schema_normalized(self, mock_fa_job_cls, base_fa):
        """A string dataset_schema is coerced to a single-element list before use
        (covers line 544: `dataset_schema = [dataset_schema]`)."""
        replies = {"node-1": _make_reply({"age": {"mean": 45.0, "count": 100}})}
        mock_fa_job_cls.return_value.execute.return_value = replies

        result = base_fa.fetch_stats("mean", dataset_schema="age")

        assert isinstance(result, FAResult)
        call_kwargs = mock_fa_job_cls.call_args.kwargs
        assert call_kwargs["dataset_schema"] == ["age"]


# ---------------------------------------------------------------------------
# TestMakeCacheKey / TestSortSchema
# ---------------------------------------------------------------------------


class TestMakeCacheKey:
    """Tests for make_cache_key stability and _sort_schema normalisation."""

    @pytest.mark.parametrize(
        "schema_a, schema_b",
        [
            # flat list: order irrelevant
            (["price", "year"], ["year", "price"]),
            # list of dicts: order irrelevant
            (
                [{"name": "age"}, {"name": "income"}],
                [{"name": "income"}, {"name": "age"}],
            ),
            # dict key order irrelevant
            ({"b": "v1", "a": "v2"}, {"a": "v2", "b": "v1"}),
            # nested list inside dict: order irrelevant
            ({"cols": ["z", "a"]}, {"cols": ["a", "z"]}),
            # deeply nested
            (
                {"group": [{"cols": ["z", "a"]}, {"cols": ["y", "b"]}]},
                {"group": [{"cols": ["b", "y"]}, {"cols": ["a", "z"]}]},
            ),
        ],
    )
    def test_equivalent_schemas_produce_same_key(self, schema_a, schema_b):
        def key(s):
            return FederatedAnalytics.make_cache_key(["node-1"], s, None)

        assert key(schema_a) == key(schema_b)

    @pytest.mark.parametrize(
        "schema_a, schema_b",
        [
            (["price"], ["year"]),
            ({"col": "age"}, {"col": "income"}),
            (["a", "b"], ["a"]),
        ],
    )
    def test_distinct_schemas_produce_different_keys(self, schema_a, schema_b):
        def key(s):
            return FederatedAnalytics.make_cache_key(["node-1"], s, None)

        assert key(schema_a) != key(schema_b)

    def test_none_schema_is_stable(self):
        k1 = FederatedAnalytics.make_cache_key(["node-1"], None, None)
        k2 = FederatedAnalytics.make_cache_key(["node-1"], None, None)
        assert k1 == k2

    def test_node_id_order_irrelevant(self):
        k1 = FederatedAnalytics.make_cache_key(["a", "b"], None, None)
        k2 = FederatedAnalytics.make_cache_key(["b", "a"], None, None)
        assert k1 == k2

    def test_different_node_ids_differ(self):
        k1 = FederatedAnalytics.make_cache_key(["node-1"], None, None)
        k2 = FederatedAnalytics.make_cache_key(["node-2"], None, None)
        assert k1 != k2

    def test_string_child_normalized_to_list(self):
        """_sort_schema normalises {key: "x"} → {key: ["x"]} so bare-string and
        single-element-list children hash identically."""
        k1 = FederatedAnalytics.make_cache_key(["node-1"], [{"demo": "col1"}], None)
        k2 = FederatedAnalytics.make_cache_key(["node-1"], [{"demo": ["col1"]}], None)
        assert k1 == k2


# ---------------------------------------------------------------------------
# TestComputableStats
# ---------------------------------------------------------------------------


class TestComputableStats:
    def test_empty_result(self):
        assert FAResult({}).computable_stats() == []

    def test_mean_count_computable(self):
        # sum+count primitives → mean, count, sum computable; variance/std require sum_sq_centered
        replies = {"n1": _make_reply({"age": {"sum": 4500.0, "count": 100}})}
        result = FAResult(replies)
        cs = result.computable_stats()
        assert "mean" in cs
        assert "count" in cs
        assert "sum" in cs
        assert "variance" not in cs
        assert "std" not in cs
        assert "histogram" not in cs

    def test_with_variance_enables_std_and_variance(self):
        # count+sum+sum_sq_centered primitives → mean, variance, std, count, sum computable
        replies = {
            "n1": _make_reply(
                {"age": {"sum_sq_centered": 396.0, "sum": 4500.0, "count": 100}}
            )
        }
        result = FAResult(replies)
        cs = result.computable_stats()
        assert "std" in cs
        assert "variance" in cs
        assert "mean" in cs
        assert "count" in cs
        assert "sum" in cs

    def test_count_only_enables_count(self):
        # count alone (without mean/variance) must not make mean/variance/sum computable
        replies = {"n1": _make_reply({"age": {"count": 100}})}
        result = FAResult(replies)
        cs = result.computable_stats()
        assert "count" in cs
        assert "mean" not in cs
        assert "variance" not in cs
        assert "sum" not in cs

    def test_histogram_computable(self):
        replies = {
            "n1": _make_reply(
                {"age": {"histogram": {"bin_edges": [0, 1], "counts": [5]}}}
            )
        }
        result = FAResult(replies)
        assert "histogram" in result.computable_stats()

    def test_histogram_and_quantile_key_enables_quantile(self):
        # quantile aggregator needs histogram + quantile keys
        replies = {
            "n1": _make_reply(
                {
                    "age": {
                        "histogram": {"bin_edges": [0, 1], "counts": [5]},
                        "quantile": [0.5],
                    }
                }
            )
        }
        result = FAResult(replies)
        cs = result.computable_stats()
        assert "quantile" in cs

    def test_result_is_sorted(self):
        # count+sum+sum_sq_centered gives multiple computable stats, verifying sort order
        replies = {
            "n1": _make_reply(
                {"age": {"sum_sq_centered": 396.0, "sum": 4500.0, "count": 100}}
            )
        }
        result = FAResult(replies)
        cs = result.computable_stats()
        assert len(cs) > 1  # ensures the test is non-trivial
        assert cs == sorted(cs)

    def test_nested_dict_schema(self):
        replies = {
            "n1": _make_reply({"tabular": {"age": {"sum": 4500.0, "count": 100}}})
        }
        result = FAResult(replies)
        cs = result.computable_stats()
        assert "mean" in cs
        assert "count" in cs
        assert "sum" in cs


# ---------------------------------------------------------------------------
# TestGlobalStats
# ---------------------------------------------------------------------------


class TestGlobalStats:
    def test_no_stat_name_returns_dict_of_all_computable(self):
        # Nodes return sum+count primitives; global_stats() aggregates all computable stats
        replies = {
            "n1": _make_reply({"age": {"sum": 4500.0, "count": 100}}),
            "n2": _make_reply({"age": {"sum": 4000.0, "count": 80}}),
        }
        result = FAResult(replies)
        all_stats = result.global_stats()
        assert isinstance(all_stats, dict)
        # top-level keys are data-tree keys, not stat names
        assert set(all_stats.keys()) == {"age"}
        assert all_stats["age"]["count"] == 180
        expected_mean = (45.0 * 100 + 50.0 * 80) / 180
        assert abs(all_stats["age"]["mean"] - expected_mean) < 1e-9

    def test_no_stat_name_empty_result_returns_empty_dict(self):
        assert FAResult({}).global_stats() == {}

    def test_derived_std_from_primitives(self):
        # std is computable from count + sum_sq_centered primitives
        replies = {
            "n1": _make_reply({"age": {"sum_sq_centered": 100.0, "count": 2}}),
            "n2": _make_reply({"age": {"sum_sq_centered": 100.0, "count": 1}}),
        }
        result = FAResult(replies)
        assert "std" in result.computable_stats()
        global_std = result.global_stats("std")
        assert isinstance(global_std, dict)
        # Σ sum_sq_centered = 200, N = 3 → variance 100 → std 10
        assert global_std["age"] == pytest.approx(10.0)

    def test_not_computable_raises(self):
        # variance is valid but missing required data when only mean+count are stored
        replies = {"n1": _make_reply({"age": {"mean": 45.0, "count": 100}})}
        result = FAResult(replies)
        with pytest.raises(FedbiomedError, match="cannot be computed"):
            result.global_stats("variance")

    def test_no_stat_name_all_stats_match_individual_calls(self):
        # sum+count primitives so mean (and sum, count) are all computable — the
        # loop must cover the derived 'mean', not just the 'count' primitive.
        replies = {
            "n1": _make_reply({"age": {"sum": 4500.0, "count": 100}}),
            "n2": _make_reply({"age": {"sum": 4000.0, "count": 80}}),
        }
        result = FAResult(replies)
        assert "mean" in result.computable_stats()  # guard: the loop covers mean
        all_at_once = result.global_stats()
        for stat in result.computable_stats():
            individual = result.global_stats(stat)
            # individual is {"age": value}; all_at_once is {"age": {stat: value, ...}}
            for col_key, col_val in individual.items():
                assert all_at_once[col_key][stat] == col_val

    def test_global_stats_scalar_leaf_raises(self):
        # Inject a bare scalar as a node output to trigger the guard in _agg.
        # computable_stats is patched to bypass the early "not computable" check
        # so that _agg is actually reached with the scalar value.
        result = FAResult.__new__(FAResult)
        result._data = {"n1": 42.0}
        with patch.object(FAResult, "computable_stats", return_value=["mean"]):
            with pytest.raises(FedbiomedError, match="Unexpected scalar"):
                result.global_stats("mean")

    def test_global_stat_no_data_raises(self):
        result = FAResult({})
        with pytest.raises(FedbiomedError, match="contains no node data"):
            result.global_stats("mean")

    def test_global_stat_sequence_output(self):
        # List-typed node outputs exercise _aggregate_tree's sequence branch
        # (covers lines 286-290: list/tuple path in _aggregate_tree)
        # Nodes return summable primitives: sum+count
        replies = {
            "n1": _make_reply(
                [{"age": {"sum": 4500.0, "count": 100}}, {"sum": 6400.0, "count": 50}]
            ),
            "n2": _make_reply(
                [{"age": {"sum": 4000.0, "count": 80}}, {"sum": 5200.0, "count": 40}]
            ),
        }
        result = FAResult(replies)
        global_mean = result.global_stats("mean")
        assert isinstance(global_mean, list)
        assert len(global_mean) == 2
        # element 0: row dict → {"age": weighted_mean}
        expected_age = (45.0 * 100 + 50.0 * 80) / 180
        assert abs(global_mean[0]["age"] - expected_age) < 1e-9
        # element 1: flat image stat → scalar
        expected_flat = (128.0 * 50 + 130.0 * 40) / 90
        assert abs(global_mean[1] - expected_flat) < 1e-9

    def test_global_stats_no_computable_stats_returns_empty_dict(self):
        # When output contains no registered stat keys, computable_stats() is empty
        # and _merge_stat_results({}) returns {} (covers line 318)
        replies = {"n1": _make_reply({"metadata": {"source": "node1"}})}
        result = FAResult(replies)
        assert result.computable_stats() == []
        assert result.global_stats() == {}

    def test_global_stats_sequence_output(self):
        # global_stats() on list-typed outputs exercises the sequence branch in
        # _merge_stat_results (covers lines 328-334)
        # sum+count primitives so the aggregated sequence carries mean as well.
        replies = {
            "n1": _make_reply(
                [{"age": {"sum": 4500.0, "count": 100}}, {"sum": 6400.0, "count": 50}]
            ),
            "n2": _make_reply(
                [{"age": {"sum": 4000.0, "count": 80}}, {"sum": 5200.0, "count": 40}]
            ),
        }
        result = FAResult(replies)
        all_stats = result.global_stats()
        assert isinstance(all_stats, list)
        assert len(all_stats) == 2
        # element 0: row dict with per-col stat map
        assert "age" in all_stats[0]
        assert "count" in all_stats[0]["age"]
        assert all_stats[0]["age"]["count"] == 180
        # element 1: flat image stat map
        assert "count" in all_stats[1]
        assert all_stats[1]["count"] == 90


# ---------------------------------------------------------------------------
# TestFilterOutput
# ---------------------------------------------------------------------------


class TestFilterOutput:
    """Tests for FAResult._filter_output and FAResult._filtered_copy."""

    # ---- _filter_output: string items ----

    def test_string_item_keeps_key(self):
        output = {
            "age": {"mean": 40.0, "count": 100},
            "weight": {"mean": 70.0, "count": 100},
        }
        result = FAResult._filter_output(output, ["age"])
        assert result == {"age": {"mean": 40.0, "count": 100}}

    def test_multiple_string_items(self):
        output = {
            "age": {"mean": 40.0, "count": 100},
            "weight": {"mean": 70.0, "count": 100},
            "height": {"mean": 170.0, "count": 100},
        }
        result = FAResult._filter_output(output, ["age", "weight"])
        assert set(result.keys()) == {"age", "weight"}
        assert "height" not in result

    # ---- _filter_output: dict items ----

    def test_dict_item_with_list_child(self):
        output = {
            "demo": {
                "col1": {"mean": 1.0, "count": 10},
                "col2": {"mean": 2.0, "count": 10},
            }
        }
        result = FAResult._filter_output(output, [{"demo": ["col1"]}])
        assert result == {"demo": {"col1": {"mean": 1.0, "count": 10}}}

    def test_dict_item_with_string_child_normalized(self):
        """Single string child is treated as a one-element list child schema."""
        output = {
            "demo": {
                "col1": {"mean": 1.0, "count": 10},
                "col2": {"mean": 2.0, "count": 10},
            }
        }
        result_str = FAResult._filter_output(output, [{"demo": "col1"}])
        result_list = FAResult._filter_output(output, [{"demo": ["col1"]}])
        assert result_str == result_list
        assert result_str == {"demo": {"col1": {"mean": 1.0, "count": 10}}}

    # ---- _filter_output: None / error cases ----

    def test_dict_item_non_dict_value_with_child_returns_none(self):
        """A child schema on a non-dict value (e.g. list) cannot be applied → returns None."""
        output = {"tag": [1, 2, 3]}
        assert FAResult._filter_output(output, [{"tag": ["x"]}]) is None

    def test_non_dict_output_returns_none(self):
        assert FAResult._filter_output([{"mean": 1.0}], ["x"]) is None

    def test_stat_leaf_output_returns_none(self):
        # A dict whose every key is in AGGREGATORS_MAP is a stat-leaf → filter returns None
        assert FAResult._filter_output({"mean": 40.0, "count": 100}, ["mean"]) is None

    def test_missing_key_returns_none(self):
        output = {"age": {"mean": 40.0, "count": 100}}
        assert FAResult._filter_output(output, ["weight"]) is None

    def test_multi_key_dict_item_returns_none(self):
        output = {"age": {"mean": 40.0}, "weight": {"mean": 70.0}}
        # A dict item with more than one key is unsupported
        assert (
            FAResult._filter_output(output, [{"age": "mean", "weight": "mean"}]) is None
        )

    def test_unsupported_item_type_returns_none(self):
        output = {"age": {"mean": 40.0}}
        assert FAResult._filter_output(output, [42]) is None

    def test_recursive_child_failure_returns_none(self):
        output = {"demo": {"col1": {"mean": 1.0, "count": 10}}}
        # "missing_col" does not exist under demo → recursive call returns None
        assert FAResult._filter_output(output, [{"demo": ["missing_col"]}]) is None

    def test_result_is_independent_deep_copy(self):
        output = {"age": {"mean": 40.0, "count": 100}}
        result = FAResult._filter_output(output, ["age"])
        result["age"]["mean"] = 999.0
        assert output["age"]["mean"] == 40.0  # original unaffected

    # ---- _filtered_copy ----

    def test_filtered_copy_returns_fa_result(self):
        replies = {
            "n1": _make_reply(
                {
                    "age": {"mean": 40.0, "count": 100},
                    "weight": {"mean": 70.0, "count": 100},
                }
            ),
            "n2": _make_reply(
                {
                    "age": {"mean": 45.0, "count": 80},
                    "weight": {"mean": 75.0, "count": 80},
                }
            ),
        }
        result = FAResult(replies)
        filtered = result._filtered_copy(["age"])
        assert isinstance(filtered, FAResult)
        assert set(filtered._data["n1"].keys()) == {"age"}
        assert set(filtered._data["n2"].keys()) == {"age"}

    def test_filtered_copy_does_not_modify_original(self):
        replies = {
            "n1": _make_reply(
                {
                    "age": {"mean": 40.0, "count": 100},
                    "weight": {"mean": 70.0, "count": 100},
                }
            )
        }
        result = FAResult(replies)
        result._filtered_copy(["age"])
        assert "weight" in result._data["n1"]

    def test_filtered_copy_returns_none_for_sequence_output(self):
        replies = {"n1": _make_reply([{"mean": 40.0, "count": 100}])}
        result = FAResult(replies)
        assert result._filtered_copy(["age"]) is None

    def test_filtered_copy_returns_none_when_key_missing(self):
        replies = {"n1": _make_reply({"age": {"mean": 40.0, "count": 100}})}
        result = FAResult(replies)
        assert result._filtered_copy(["weight"]) is None

    def test_filtered_copy_one_node_failure_returns_none(self):
        """If any node's output cannot be filtered, filtered_copy returns None."""
        replies = {
            "n1": _make_reply({"age": {"mean": 40.0, "count": 100}}),
            "n2": _make_reply({"weight": {"mean": 70.0, "count": 100}}),  # no "age"
        }
        result = FAResult(replies)
        assert result._filtered_copy(["age"]) is None

    def test_filtered_copy_empty_result(self):
        """An empty schema list produces empty dicts per node (no keys requested)."""
        replies = {"n1": _make_reply({"age": {"mean": 40.0, "count": 100}})}
        result = FAResult(replies)
        filtered = result._filtered_copy([])
        assert isinstance(filtered, FAResult)
        assert filtered._data["n1"] == {}


# ---------------------------------------------------------------------------
# TestCacheFallback
# ---------------------------------------------------------------------------


class TestCacheFallback:
    """Tests for the superset-schema cache fallback in fetch_stats."""

    @patch("fedbiomed.researcher.federated_workflows._federated_analytics.FARequestJob")
    def test_none_schema_serves_subset_request(self, mock_fa_job_cls, base_fa):
        """A None-schema cached result is reused (filtered) for a specific schema request."""
        replies = {
            "node-1": _make_reply(
                {
                    "age": {"mean": 40.0, "count": 100},
                    "weight": {"mean": 70.0, "count": 100},
                }
            ),
            "node-2": _make_reply(
                {
                    "age": {"mean": 45.0, "count": 80},
                    "weight": {"mean": 75.0, "count": 80},
                }
            ),
        }
        mock_fa_job_cls.return_value.execute.return_value = replies

        # First call: None schema — populates cache
        base_fa.fetch_stats("mean")
        assert mock_fa_job_cls.call_count == 1

        # Second call: specific schema — should be served from cache without a new request
        result = base_fa.fetch_stats("mean", dataset_schema=["age"])
        assert mock_fa_job_cls.call_count == 1  # no additional request
        assert isinstance(result, FAResult)
        assert set(result._data["node-1"].keys()) == {"age"}

    @patch("fedbiomed.researcher.federated_workflows._federated_analytics.FARequestJob")
    def test_larger_explicit_schema_serves_subset(self, mock_fa_job_cls, base_fa):
        """A cached result for a superset schema is reused for a subset schema request."""
        replies = {
            "node-1": _make_reply(
                {
                    "age": {"mean": 40.0, "count": 100},
                    "weight": {"mean": 70.0, "count": 100},
                }
            ),
            "node-2": _make_reply(
                {
                    "age": {"mean": 45.0, "count": 80},
                    "weight": {"mean": 75.0, "count": 80},
                }
            ),
        }
        mock_fa_job_cls.return_value.execute.return_value = replies

        base_fa.fetch_stats("mean", dataset_schema=["age", "weight"])
        assert mock_fa_job_cls.call_count == 1

        result = base_fa.fetch_stats("mean", dataset_schema=["age"])
        assert mock_fa_job_cls.call_count == 1  # fallback used, no new request
        assert set(result._data["node-1"].keys()) == {"age"}

    @patch("fedbiomed.researcher.federated_workflows._federated_analytics.FARequestJob")
    def test_fallback_result_stored_in_cache(self, mock_fa_job_cls, base_fa):
        """The filtered result is stored so a third call with the same schema costs nothing."""
        replies = {
            "node-1": _make_reply(
                {
                    "age": {"mean": 40.0, "count": 100},
                    "weight": {"mean": 70.0, "count": 100},
                }
            ),
            "node-2": _make_reply(
                {
                    "age": {"mean": 45.0, "count": 80},
                    "weight": {"mean": 75.0, "count": 80},
                }
            ),
        }
        mock_fa_job_cls.return_value.execute.return_value = replies

        base_fa.fetch_stats("mean")
        base_fa.fetch_stats("mean", dataset_schema=["age"])  # fallback
        base_fa.fetch_stats("mean", dataset_schema=["age"])  # should hit cache directly
        assert mock_fa_job_cls.call_count == 1

    @patch("fedbiomed.researcher.federated_workflows._federated_analytics.FARequestJob")
    def test_fallback_skipped_when_superset_lacks_stat(self, mock_fa_job_cls, base_fa):
        """Fallback is not used when the cached result does not have all requested stats."""
        replies_count_only = {
            "node-1": _make_reply({"age": {"count": 100}, "weight": {"count": 100}}),
            "node-2": _make_reply({"age": {"count": 80}, "weight": {"count": 80}}),
        }
        replies_mean = {
            "node-1": _make_reply({"age": {"mean": 40.0, "count": 100}}),
            "node-2": _make_reply({"age": {"mean": 45.0, "count": 80}}),
        }
        mock_fa_job_cls.return_value.execute.side_effect = [
            replies_count_only,
            replies_mean,
        ]

        base_fa.fetch_stats("count")  # cached result has no "mean"
        base_fa.fetch_stats("mean", dataset_schema=["age"])  # must issue a new request
        assert mock_fa_job_cls.call_count == 2

    @patch("fedbiomed.researcher.federated_workflows._federated_analytics.FARequestJob")
    def test_fallback_skipped_when_schema_is_none(self, mock_fa_job_cls, base_fa):
        """Fallback scan is only triggered when dataset_schema is not None."""
        replies = {
            "node-1": _make_reply({"age": {"mean": 40.0, "count": 100}}),
            "node-2": _make_reply({"age": {"mean": 45.0, "count": 80}}),
        }
        mock_fa_job_cls.return_value.execute.return_value = replies

        # Two None-schema calls must each hit cache the second time — not fallback
        base_fa.fetch_stats("mean")
        base_fa.fetch_stats("mean")  # exact cache hit
        assert mock_fa_job_cls.call_count == 1  # only one network request

    @patch("fedbiomed.researcher.federated_workflows._federated_analytics.FARequestJob")
    def test_fallback_skipped_when_partial_cache_exists(self, mock_fa_job_cls, base_fa):
        """Fallback is skipped when a partial (non-None) cached entry already exists.

        Sequence: build a partial cache via a direct network request (no superset in cache
        yet), then add a superset, then request a missing stat — the partial cache entry
        (cached is not None) prevents fallback from triggering, so a merge request is sent.
        """
        replies_count_age = {
            "node-1": _make_reply({"age": {"count": 100}}),
            "node-2": _make_reply({"age": {"count": 80}}),
        }
        replies_superset = {
            "node-1": _make_reply(
                {
                    "age": {"mean": 40.0, "count": 100},
                    "weight": {"mean": 70.0, "count": 100},
                }
            ),
            "node-2": _make_reply(
                {
                    "age": {"mean": 45.0, "count": 80},
                    "weight": {"mean": 75.0, "count": 80},
                }
            ),
        }
        replies_mean_age = {  # a 'mean' request returns sum+count primitives
            "node-1": _make_reply({"age": {"sum": 4000.0, "count": 100}}),
            "node-2": _make_reply({"age": {"sum": 3600.0, "count": 80}}),
        }
        mock_fa_job_cls.return_value.execute.side_effect = [
            replies_count_age,  # step 1: partial cache for ["age"] (no superset yet → network)
            replies_superset,  # step 2: superset cached under None schema
            replies_mean_age,  # step 3: merge "mean" into ["age"] partial (fallback skipped)
        ]

        # Step 1: populate partial cache for ["age"] schema (no superset in cache → network request)
        base_fa.fetch_stats("count", dataset_schema=["age"])
        assert mock_fa_job_cls.call_count == 1

        # Step 2: populate superset under None schema
        base_fa.fetch_stats("mean")
        assert mock_fa_job_cls.call_count == 2

        # Step 3: request "mean" for ["age"] — partial cache exists (not None), so fallback is
        # skipped and a merge network request is made instead
        result = base_fa.fetch_stats("mean", dataset_schema=["age"])
        assert mock_fa_job_cls.call_count == 3
        assert "mean" in result.computable_stats()  # derived from merged sum+count

    @patch("fedbiomed.researcher.federated_workflows._federated_analytics.FARequestJob")
    def test_fallback_skipped_for_sequence_output(self, mock_fa_job_cls, base_fa):
        """filtered_copy returns None for sequence outputs — fallback is not used."""
        replies_seq = {
            "node-1": _make_reply(
                [{"mean": 40.0, "count": 100}, {"mean": 128.0, "count": 50}]
            ),
            "node-2": _make_reply(
                [{"mean": 45.0, "count": 80}, {"mean": 130.0, "count": 40}]
            ),
        }
        replies_schema = {
            "node-1": _make_reply({"age": {"mean": 40.0, "count": 100}}),
            "node-2": _make_reply({"age": {"mean": 45.0, "count": 80}}),
        }
        mock_fa_job_cls.return_value.execute.side_effect = [replies_seq, replies_schema]

        base_fa.fetch_stats("mean")  # sequence output cached under None schema
        base_fa.fetch_stats(
            "mean", dataset_schema=["age"]
        )  # filtered_copy fails → new request
        assert mock_fa_job_cls.call_count == 2


# ---------------------------------------------------------------------------
# TestSecaggIntegration — encrypted-path tests for FederatedAnalytics
# ---------------------------------------------------------------------------


def _make_encrypted_reply(params_encrypted, output_schema) -> MagicMock:
    """Create a MagicMock FAReply representing the encrypted path."""
    r = MagicMock(spec=FAReply)
    r.encrypted = True
    r.params_encrypted = params_encrypted
    r.output_schema = output_schema
    r.output = None
    return r


class TestSecaggIntegration:
    @pytest.fixture
    def mock_secagg(self):
        secagg = MagicMock()
        secagg.active = True
        secagg.train_arguments.return_value = {
            "secagg_scheme": "LOM",
            "secagg_random": 0.5,
            "secagg_clipping_range": SAParameters.FA_CLIPPING_RANGE,
            "parties": ["node-1", "node-2"],
        }
        return secagg

    @pytest.fixture
    def secagg_fa(self, mock_fds, mock_requests, mock_secagg):
        return FederatedAnalytics(
            fds=mock_fds,
            experiment_id="exp-secagg",
            researcher_id="res-456",
            reqs=mock_requests,
            experimentation_folder="/tmp/fedbiomed",
            secagg=mock_secagg,
        )

    def test_secagg_with_fewer_than_two_nodes_raises(
        self, secagg_fa, mock_fds, mock_secagg
    ):
        """The <2-node guard lives in SecureAggregation.setup; FA delegates and surfaces it.

        FA calls secagg.setup() with the participating nodes; if setup rejects them
        (the centralised guard), the error propagates out of fetch_stats.
        """
        mock_fds.node_ids.return_value = ["node-1"]
        mock_secagg.setup.side_effect = FedbiomedSecureAggregationError(
            "Secure aggregation requires at least 2 nodes"
        )
        with pytest.raises(FedbiomedError, match="at least 2 nodes"):
            secagg_fa.fetch_stats("mean")
        mock_secagg.setup.assert_called_once()

    def test_secagg_setup_sets_fa_clipping_range_on_scheme(
        self, secagg_fa, mock_secagg
    ):
        """FA sends both fixed ranges so the nodes can validate them."""
        secagg_arguments = secagg_fa._secagg_setup(["node-1", "node-2"])
        assert mock_secagg.clipping_range == SAParameters.FA_CLIPPING_RANGE
        assert (
            secagg_arguments["secagg_clipping_range"] == SAParameters.FA_CLIPPING_RANGE
        )
        assert secagg_arguments["secagg_target_range"] == SAParameters.FA_TARGET_RANGE

    @patch("fedbiomed.researcher.federated_workflows._federated_analytics.FARequestJob")
    def test_aggregate_uses_fa_ranges(self, mock_fa_job_cls, secagg_fa, mock_secagg):
        """FA sets the clipping range on the scheme and passes the target range to aggregate()."""
        schema = [["x"]]
        mock_fa_job_cls.return_value.execute.return_value = {
            "node-1": _make_encrypted_reply([1], schema),
            "node-2": _make_encrypted_reply([1], schema),
        }
        mock_secagg.aggregate.return_value = [1.0]
        secagg_fa.fetch_stats("mean")
        assert mock_secagg.clipping_range == SAParameters.FA_CLIPPING_RANGE
        assert (
            mock_secagg.aggregate.call_args.kwargs["target_range"]
            == SAParameters.FA_TARGET_RANGE
        )

    @patch("fedbiomed.researcher.federated_workflows._federated_analytics.FARequestJob")
    def test_encrypted_replies_are_decrypted(
        self, mock_fa_job_cls, secagg_fa, mock_secagg
    ):
        """Encrypted replies trigger secagg.aggregate(); result is unflattened into FAResult."""
        # output_schema encodes {"age": {"sum": _, "count": _}}
        schema = [["age", "sum"], ["age", "count"]]
        reply_n1 = _make_encrypted_reply([100, 200], schema)
        reply_n2 = _make_encrypted_reply([150, 300], schema)
        mock_fa_job_cls.return_value.execute.return_value = {
            "node-1": reply_n1,
            "node-2": reply_n2,
        }

        # aggregate() returns the per-node average; _execute_and_update_cache
        # rescales by num_nodes (2 here) to recover the additive sum.
        mock_secagg.aggregate.return_value = [45.0, 180]

        result = secagg_fa.fetch_stats("mean")

        mock_secagg.aggregate.assert_called_once()
        assert isinstance(result, FAResult)
        # The aggregated output is stored under "__secagg__"
        assert "__secagg__" in result.node_ids
        node_output = result.node_stats("__secagg__")
        assert node_output == {"age": {"sum": 90.0, "count": 360}}

    @patch("fedbiomed.researcher.federated_workflows._federated_analytics.FARequestJob")
    def test_secagg_setup_called_with_node_ids(
        self, mock_fa_job_cls, secagg_fa, mock_secagg, mock_fds
    ):
        """_secagg_setup calls secagg.setup() with the node IDs from the federated dataset."""
        schema = [["x"]]
        reply = _make_encrypted_reply([1], schema)
        mock_fa_job_cls.return_value.execute.return_value = {"node-1": reply}
        mock_fds.node_ids.return_value = ["node-1", "node-2"]
        mock_secagg.aggregate.return_value = [5.0]

        secagg_fa.fetch_stats("mean")

        mock_secagg.setup.assert_called_once_with(
            parties=["node-1", "node-2"],
            experiment_id="exp-secagg",
            researcher_id="res-456",
            insecure_validation=False,
        )

    @patch("fedbiomed.researcher.federated_workflows._federated_analytics.FARequestJob")
    def test_fa_round_counter_increments(self, mock_fa_job_cls, secagg_fa, mock_secagg):
        """fa_round injected into secagg_arguments increments with each FA request."""
        schema = [["x"]]
        mock_fa_job_cls.return_value.execute.return_value = {
            "node-1": _make_encrypted_reply([1], schema)
        }
        mock_secagg.aggregate.return_value = [1.0]

        secagg_fa.fetch_stats("count")
        secagg_fa.fetch_stats("mean")  # triggers second request (mean is missing)

        assert secagg_fa._fa_round_counter == 2
        calls = mock_fa_job_cls.call_args_list
        assert calls[0].kwargs["secagg_arguments"]["fa_round"] == 1
        assert calls[1].kwargs["secagg_arguments"]["fa_round"] == 2

    @patch("fedbiomed.researcher.federated_workflows._federated_analytics.FARequestJob")
    def test_secagg_arguments_forwarded_to_fa_job(
        self, mock_fa_job_cls, secagg_fa, mock_secagg
    ):
        """train_arguments() dict (plus fa_round) is passed verbatim to FARequestJob."""
        schema = [["x"]]
        mock_fa_job_cls.return_value.execute.return_value = {
            "node-1": _make_encrypted_reply([1], schema)
        }
        mock_secagg.aggregate.return_value = [1.0]

        secagg_fa.fetch_stats("count")

        kwargs = mock_fa_job_cls.call_args.kwargs
        sa = kwargs["secagg_arguments"]
        assert sa["secagg_scheme"] == "LOM"
        assert sa["fa_round"] == 1

    @patch("fedbiomed.researcher.federated_workflows._federated_analytics.FARequestJob")
    def test_inactive_secagg_sends_no_secagg_arguments(
        self, mock_fa_job_cls, mock_fds, mock_requests
    ):
        """When secagg is inactive, secagg_arguments=None is forwarded to FARequestJob."""
        from fedbiomed.researcher.secagg import SecureAggregation

        fa = FederatedAnalytics(
            fds=mock_fds,
            experiment_id="exp",
            researcher_id="res",
            reqs=mock_requests,
            experimentation_folder="/tmp",
            secagg=SecureAggregation(active=False),
        )
        mock_fa_job_cls.return_value.execute.return_value = {
            "node-1": _make_reply({"x": {"count": 10}})
        }

        fa.fetch_stats("count")

        kwargs = mock_fa_job_cls.call_args.kwargs
        assert kwargs["secagg_arguments"] is None

    @patch("fedbiomed.researcher.federated_workflows._federated_analytics.FARequestJob")
    def test_encrypted_merge_accumulates_stats(
        self, mock_fa_job_cls, secagg_fa, mock_secagg
    ):
        """Two sequential encrypted requests merge their aggregate outputs in FAResult."""
        schema_count = [["x", "count"]]
        schema_sum = [["x", "sum"]]
        replies_count = {"node-1": _make_encrypted_reply([100], schema_count)}
        replies_sum = {"node-1": _make_encrypted_reply([500], schema_sum)}
        mock_fa_job_cls.return_value.execute.side_effect = [replies_count, replies_sum]
        mock_secagg.aggregate.side_effect = [[100.0], [500.0]]

        secagg_fa.fetch_stats("count")
        secagg_fa.fetch_stats("mean")  # mean requires sum + count

        output = secagg_fa._results_store[
            list(secagg_fa._results_store.keys())[-1]
        ].node_stats("__secagg__")
        assert output["x"]["count"] == 100.0
        assert output["x"]["sum"] == 500.0
