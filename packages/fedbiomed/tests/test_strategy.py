import pytest
from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedStrategyError
from fedbiomed.researcher.strategies.default_strategy import DefaultStrategy


@pytest.fixture
def default_strategy():
    return DefaultStrategy()


@pytest.fixture
def sample_training_replies():
    return {
        "node-1": {
            "success": True,
            "node_id": "node-1",
            "params": {"w": [1]},
            "sample_size": 100,
            "encryption_factor": [0.5],
        },
        "node-2": {
            "success": True,
            "node_id": "node-2",
            "params": {"w": [2]},
            "sample_size": 300,
        },
    }


def test_default_strategy_01_sample_nodes_returns_all(default_strategy):
    nodes = ["node-1", "node-2"]
    sampled = default_strategy.sample_nodes(nodes, 0)
    assert sampled == nodes
    assert default_strategy._sampling_node_history[0] == nodes


def test_default_strategy_02_refine_successful_replies(
    default_strategy,
    sample_training_replies,
):
    default_strategy._sampling_node_history[0] = ["node-1", "node-2"]
    (model_params, weights, total_rows, encryption_factors) = default_strategy.refine(
        sample_training_replies, 0
    )
    assert model_params == {"node-1": {"w": [1]}, "node-2": {"w": [2]}}
    assert weights == {"node-1": 0.25, "node-2": 0.75}
    assert total_rows == 400
    assert encryption_factors == {"node-1": [0.5], "node-2": None}


def test_default_strategy_03_refine_missing_node_reply_raises(
    default_strategy,
    sample_training_replies,
):
    default_strategy._sampling_node_history[0] = ["node-1", "node-2"]
    training_replies = sample_training_replies
    training_replies.pop("node-2")
    with pytest.raises(FedbiomedStrategyError, match=ErrorNumbers.FB408.value):
        default_strategy.refine(training_replies, 0)


def test_default_strategy_04_refine_successful_node_reply_missing_sample_size_raises(
    default_strategy,
    sample_training_replies,
):
    default_strategy._sampling_node_history[0] = ["node-1", "node-2"]
    training_replies = sample_training_replies
    training_replies["node-2"].pop("sample_size")
    with pytest.raises(FedbiomedStrategyError, match=ErrorNumbers.FB402.value):
        default_strategy.refine(training_replies, 0)


def test_default_strategy_05_refine_successful_node_reply_sample_size_is_none_raises(
    default_strategy,
    sample_training_replies,
):
    default_strategy._sampling_node_history[0] = ["node-1", "node-2"]
    training_replies = sample_training_replies
    training_replies["node-2"]["sample_size"] = None
    with pytest.raises(FedbiomedStrategyError, match=ErrorNumbers.FB402.value):
        default_strategy.refine(training_replies, 0)


def test_default_strategy_06_refine_unsuccessful_node_reply_raises(
    default_strategy,
    sample_training_replies,
):
    default_strategy._sampling_node_history[0] = ["node-1", "node-2"]
    training_replies = sample_training_replies
    training_replies["node-2"] = {
        "success": False,
        "node_id": "node-2",
        "msg": "Error message",
    }
    with pytest.raises(FedbiomedStrategyError, match=ErrorNumbers.FB402.value):
        default_strategy.refine(training_replies, 0)
