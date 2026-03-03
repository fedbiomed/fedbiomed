import uuid
from unittest.mock import MagicMock, patch

import pytest

from fedbiomed.common.exceptions import FedbiomedError, FedbiomedExperimentError
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
    """Create a MagicMock FAReply with the given output dict."""
    r = MagicMock(spec=FAReply)
    r.output = output
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

    # --- has_stat ---

    def test_has_stat_true_row(self):
        # ROW: {col: {stat: val}}
        replies = {"n1": _make_reply({"age": {"mean": 45.0, "count": 100}})}
        result = FAResult(replies)
        assert result.has_stat("mean") is True
        assert result.has_stat("count") is True

    def test_has_stat_false_missing(self):
        replies = {"n1": _make_reply({"age": {"mean": 45.0, "count": 100}})}
        result = FAResult(replies)
        assert result.has_stat("variance") is False

    def test_has_stat_false_partial_columns(self):
        # mean in one column but not another → has_stat should be False
        replies = {
            "n1": _make_reply(
                {"age": {"mean": 45.0, "count": 100}, "weight": {"count": 100}}
            ),
        }
        result = FAResult(replies)
        assert result.has_stat("mean") is False
        assert result.has_stat("count") is True

    def test_has_stat_false_partial_nodes(self):
        # mean missing from one node → has_stat should be False
        replies = {
            "n1": _make_reply({"age": {"mean": 45.0, "count": 100}}),
            "n2": _make_reply({"age": {"count": 80}}),
        }
        result = FAResult(replies)
        assert result.has_stat("mean") is False
        assert result.has_stat("count") is True

    def test_has_stat_image_flat(self):
        # IMAGE: {stat: val}
        replies = {"n1": _make_reply({"mean": 128.0, "count": 100})}
        result = FAResult(replies)
        assert result.has_stat("mean") is True
        assert result.has_stat("variance") is False

    def test_has_stat_nested_dict_schema(self):
        # Nested: {key: {col: {stat: val}}}
        replies = {
            "n1": _make_reply({"tabular": {"age": {"mean": 45.0, "count": 100}}}),
        }
        result = FAResult(replies)
        assert result.has_stat("mean") is True
        assert result.has_stat("min") is False

    def test_has_stat_sequence_schema(self):
        # Tuple schema: ({col: {stat}}, {stat: val})
        replies = {
            "n1": _make_reply(
                ({"age": {"mean": 45.0, "count": 100}}, {"mean": 128.0, "count": 50})
            ),
        }
        result = FAResult(replies)
        assert result.has_stat("mean") is True
        assert result.has_stat("variance") is False

    def test_has_stat_empty_returns_false(self):
        assert FAResult({}).has_stat("mean") is False

    # --- available_stats ---

    def test_available_stats_row(self):
        replies = {"n1": _make_reply({"age": {"mean": 45.0, "count": 100}})}
        result = FAResult(replies)
        assert sorted(result.available_stats()) == ["count", "mean"]

    def test_available_stats_image_flat(self):
        replies = {"n1": _make_reply({"min": 0.0, "max": 255.0})}
        result = FAResult(replies)
        assert sorted(result.available_stats()) == ["max", "min"]

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
        replies = {
            "n1": _make_reply({"age": {"mean": 45.0, "count": 100}}),
            "n2": _make_reply({"age": {"mean": 50.0, "count": 80}}),
        }
        result = FAResult(replies)
        global_mean = result.global_stat("mean")
        expected_age = (45.0 * 100 + 50.0 * 80) / 180
        assert isinstance(global_mean, dict)
        assert abs(global_mean["age"] - expected_age) < 1e-9

    def test_global_stats_mean_image_flat(self):
        # IMAGE: {stat: val} — result is a scalar directly
        replies = {
            "n1": _make_reply({"mean": 128.0, "count": 100}),
            "n2": _make_reply({"mean": 130.0, "count": 80}),
        }
        result = FAResult(replies)
        global_mean = result.global_stat("mean")
        expected = (128.0 * 100 + 130.0 * 80) / 180
        assert abs(global_mean - expected) < 1e-9

    def test_global_stats_min_row(self):
        replies = {
            "n1": _make_reply({"age": {"min": 20.0}}),
            "n2": _make_reply({"age": {"min": 18.0}}),
        }
        result = FAResult(replies)
        global_min = result.global_stat("min")
        assert global_min == {"age": 18.0}

    def test_global_stats_max_row(self):
        replies = {
            "n1": _make_reply({"age": {"max": 80.0}}),
            "n2": _make_reply({"age": {"max": 90.0}}),
        }
        result = FAResult(replies)
        assert result.global_stat("max") == {"age": 90.0}

    def test_global_stats_count_row(self):
        replies = {
            "n1": _make_reply({"age": {"count": 100}}),
            "n2": _make_reply({"age": {"count": 80}}),
        }
        result = FAResult(replies)
        assert result.global_stat("count") == {"age": 180}

    def test_global_stats_nested_dict_schema(self):
        # Nested: {key: {col: {stat: val}}} — result is {key: {col: val}}
        replies = {
            "n1": _make_reply({"tabular": {"age": {"mean": 45.0, "count": 100}}}),
            "n2": _make_reply({"tabular": {"age": {"mean": 50.0, "count": 80}}}),
        }
        result = FAResult(replies)
        global_mean = result.global_stat("mean")
        expected_age = (45.0 * 100 + 50.0 * 80) / 180
        assert isinstance(global_mean, dict)
        assert abs(global_mean["tabular"]["age"] - expected_age) < 1e-9

    def test_global_stats_missing_stat_raises(self):
        replies = {"n1": _make_reply({"age": {"mean": 45.0, "count": 100}})}
        result = FAResult(replies)
        with pytest.raises(FedbiomedError, match="not computable"):
            result.global_stat("variance")

    def test_global_stats_unregistered_stat_raises(self):
        # A stat key exists in the data but has no registered aggregator;
        # no leaf dict is detected so the stat is not computable
        replies = {"n1": _make_reply({"skewness": 0.5})}
        result = FAResult(replies)
        with pytest.raises(FedbiomedError, match="not computable"):
            result.global_stat("skewness")

    # --- merge ---

    def test_merge_adds_new_stat(self):
        result = FAResult({"n1": _make_reply({"age": {"mean": 45.0, "count": 100}})})
        new_replies = {
            "n1": _make_reply({"age": {"variance": 4.0, "mean": 45.0, "count": 100}})
        }
        result.merge(new_replies)
        assert result.has_stat("variance")
        assert result.has_stat("mean")

    def test_merge_preserves_existing_stats(self):
        result = FAResult({"n1": _make_reply({"age": {"mean": 45.0, "count": 100}})})
        new_replies = {"n1": _make_reply({"age": {"min": 18.0}})}
        result.merge(new_replies)
        assert result.has_stat("mean")
        assert result.has_stat("min")

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

    def test_deep_merge_incompatible_types_raises(self):
        with pytest.raises(FedbiomedError):
            FAResult._deep_merge({"a": 1}, [1])

        with pytest.raises(FedbiomedError):
            FAResult._deep_merge([1], {"a": 1})

        with pytest.raises(FedbiomedError):
            # One sequence, one scalar
            FAResult._deep_merge([1], 1)

    # --- all_node_stats ---

    def test_all_node_stats_returns_dict_by_default(self):
        replies = {
            "n1": _make_reply({"age": {"mean": 45.0, "count": 100}}),
            "n2": _make_reply({"age": {"mean": 50.0, "count": 80}}),
        }
        result = FAResult(replies)
        all_stats = result.all_node_stats()
        assert isinstance(all_stats, dict)
        assert set(all_stats.keys()) == {"n1", "n2"}
        assert all_stats["n1"] == result.node_stats("n1")
        assert all_stats["n2"] == result.node_stats("n2")

    def test_all_node_stats_empty_result(self):
        result = FAResult({})
        assert result.all_node_stats() == {}


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
    def test_compute_analytics_returns_fa_result(self, mock_fa_job_cls, base_fa):
        replies = {"n1": _make_reply({"age": {"mean": 45.0, "count": 100}})}
        mock_fa_job_cls.return_value.execute.return_value = (replies, {})

        result = base_fa.compute_analytics("mean", fa_args={})

        assert isinstance(result, FAResult)
        mock_fa_job_cls.assert_called_once()

    @patch("fedbiomed.researcher.federated_workflows._federated_analytics.FARequestJob")
    def test_mean_returns_fa_result(self, mock_fa_job_cls, base_fa):
        replies = {"node-1": _make_reply({"age": {"mean": 45.0, "count": 100}})}
        mock_fa_job_cls.return_value.execute.return_value = (replies, {})

        result = base_fa.mean()

        assert isinstance(result, FAResult)
        assert result.has_stat("mean")

    # --- Caching tests ---

    @patch("fedbiomed.researcher.federated_workflows._federated_analytics.FARequestJob")
    def test_same_stat_same_args_uses_cache(self, mock_fa_job_cls, base_fa):
        """Second call with same stat and args must not trigger a new FARequestJob."""
        replies = {"node-1": _make_reply({"age": {"mean": 45.0, "count": 100}})}
        mock_fa_job_cls.return_value.execute.return_value = (replies, {})

        result1 = base_fa.mean()
        result2 = base_fa.mean()

        assert mock_fa_job_cls.call_count == 1
        assert result1 is result2

    @patch("fedbiomed.researcher.federated_workflows._federated_analytics.FARequestJob")
    def test_different_args_bypass_cache(self, mock_fa_job_cls, base_fa):
        """Different fa_args must cause a separate network request."""
        replies = {"node-1": _make_reply({"age": {"mean": 45.0, "count": 100}})}
        mock_fa_job_cls.return_value.execute.return_value = (replies, {})

        base_fa.compute_analytics("mean", fa_args={"key": "a"})
        base_fa.compute_analytics("mean", fa_args={"key": "b"})

        assert mock_fa_job_cls.call_count == 2

    @patch("fedbiomed.researcher.federated_workflows._federated_analytics.FARequestJob")
    def test_cached_dependency_avoids_request(self, mock_fa_job_cls, base_fa):
        """
        After requesting variance (which includes mean+count as dependencies),
        a subsequent mean() with the same args must be served from cache.
        """
        variance_replies = {
            "node-1": _make_reply(
                {"age": {"variance": 4.0, "mean": 45.0, "count": 100}}
            )
        }
        mock_fa_job_cls.return_value.execute.return_value = (variance_replies, {})

        base_fa.variance()
        assert mock_fa_job_cls.call_count == 1

        result = base_fa.mean()
        assert mock_fa_job_cls.call_count == 1  # no new network call

        assert result.has_stat("mean")
        assert result.has_stat("variance")

    @patch("fedbiomed.researcher.federated_workflows._federated_analytics.FARequestJob")
    def test_only_missing_stats_requested(self, mock_fa_job_cls, base_fa):
        """When some stats are cached, only the missing ones are sent to nodes."""
        mean_replies = {"node-1": _make_reply({"age": {"mean": 45.0, "count": 100}})}
        mock_fa_job_cls.return_value.execute.return_value = (mean_replies, {})
        base_fa.mean()

        min_replies = {"node-1": _make_reply({"age": {"min": 18.0}})}
        mock_fa_job_cls.return_value.execute.return_value = (min_replies, {})
        result = base_fa.min()

        assert mock_fa_job_cls.call_count == 2
        second_call_kwargs = mock_fa_job_cls.call_args_list[1].kwargs
        assert second_call_kwargs["stats"] == ["min"]

        assert result.has_stat("mean")
        assert result.has_stat("min")

    @patch("fedbiomed.researcher.federated_workflows._federated_analytics.FARequestJob")
    def test_compute_multiple_stats_at_once(self, mock_fa_job_cls, base_fa):
        replies = {
            "node-1": _make_reply({"age": {"mean": 45.0, "count": 100, "min": 18.0}})
        }
        mock_fa_job_cls.return_value.execute.return_value = (replies, {})

        result = base_fa.compute_analytics(stats=["mean", "min"])

        assert result.has_stat("mean")
        assert result.has_stat("min")
        assert mock_fa_job_cls.call_count == 1

    @patch("fedbiomed.researcher.federated_workflows._federated_analytics.FARequestJob")
    def test_node_change_invalidates_cache(self, mock_fa_job_cls, base_fa, mock_fds):
        """Adding or removing a node creates a new cache entry."""
        replies = {"node-1": _make_reply({"age": {"mean": 45.0, "count": 100}})}
        mock_fa_job_cls.return_value.execute.return_value = (replies, {})

        base_fa.mean()
        assert mock_fa_job_cls.call_count == 1

        # Simulate a node being added to the federation
        mock_fds.node_ids.return_value = ["node-1", "node-2", "node-3"]
        replies_3 = {
            "node-1": _make_reply({"age": {"mean": 45.0, "count": 100}}),
            "node-3": _make_reply({"age": {"mean": 48.0, "count": 60}}),
        }
        mock_fa_job_cls.return_value.execute.return_value = (replies_3, {})

        base_fa.mean()
        assert mock_fa_job_cls.call_count == 2  # new node set → cache miss


# ---------------------------------------------------------------------------
# TestComputableStats
# ---------------------------------------------------------------------------


class TestComputableStats:
    def test_empty_result(self):
        assert FAResult({}).computable_stats() == []

    def test_mean_count_computable(self):
        # mean+count in data → mean, count, sum are computable; variance/std/min/max are not
        replies = {"n1": _make_reply({"age": {"mean": 45.0, "count": 100}})}
        result = FAResult(replies)
        cs = result.computable_stats()
        assert "mean" in cs
        assert "count" in cs
        assert "sum" in cs
        assert "variance" not in cs
        assert "std" not in cs
        assert "min" not in cs

    def test_with_variance_enables_std_and_variance(self):
        replies = {
            "n1": _make_reply({"age": {"mean": 45.0, "count": 100, "variance": 4.0}})
        }
        result = FAResult(replies)
        cs = result.computable_stats()
        assert "std" in cs
        assert "variance" in cs
        assert "mean" in cs
        assert "count" in cs
        assert "sum" in cs

    def test_min_only_enables_min(self):
        replies = {"n1": _make_reply({"age": {"min": 18.0}})}
        result = FAResult(replies)
        cs = result.computable_stats()
        assert "min" in cs
        assert "max" not in cs
        assert "mean" not in cs

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
        replies = {
            "n1": _make_reply(
                {"age": {"mean": 45.0, "count": 100, "min": 18.0, "max": 90.0}}
            )
        }
        result = FAResult(replies)
        cs = result.computable_stats()
        assert cs == sorted(cs)

    def test_nested_dict_schema(self):
        replies = {
            "n1": _make_reply({"tabular": {"age": {"mean": 45.0, "count": 100}}})
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
        replies = {
            "n1": _make_reply({"age": {"min": 20.0, "max": 80.0}}),
            "n2": _make_reply({"age": {"min": 18.0, "max": 90.0}}),
        }
        result = FAResult(replies)
        all_stats = result.global_stats()
        assert isinstance(all_stats, dict)
        # top-level keys are data-tree keys, not stat names
        assert set(all_stats.keys()) == {"age"}
        assert all_stats["age"]["min"] == 18.0
        assert all_stats["age"]["max"] == 90.0

    def test_no_stat_name_empty_result_returns_empty_dict(self):
        assert FAResult({}).global_stats() == {}

    def test_derived_std_from_mean_variance_count(self):
        # std is not a stored key but is computable from mean+variance+count
        replies = {
            "n1": _make_reply({"age": {"mean": 45.0, "variance": 4.0, "count": 100}}),
            "n2": _make_reply({"age": {"mean": 50.0, "variance": 9.0, "count": 80}}),
        }
        result = FAResult(replies)
        assert "std" in result.computable_stats()
        global_std = result.global_stat("std")
        assert isinstance(global_std, dict)
        assert "age" in global_std
        assert global_std["age"] > 0

    def test_not_computable_raises(self):
        # variance is not computable when only mean+count are stored
        replies = {"n1": _make_reply({"age": {"mean": 45.0, "count": 100}})}
        result = FAResult(replies)
        with pytest.raises(FedbiomedError, match="not computable"):
            result.global_stat("variance")

    def test_no_stat_name_all_stats_match_individual_calls(self):
        replies = {
            "n1": _make_reply({"age": {"mean": 45.0, "count": 100}}),
            "n2": _make_reply({"age": {"mean": 50.0, "count": 80}}),
        }
        result = FAResult(replies)
        all_at_once = result.global_stats()
        for stat in result.computable_stats():
            individual = result.global_stat(stat)
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
                result.global_stat("mean")

    def test_global_stat_no_data_raises(self):
        result = FAResult({})
        with pytest.raises(FedbiomedError, match="contains no node data"):
            result.global_stat("mean")
