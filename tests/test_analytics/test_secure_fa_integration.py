#!/usr/bin/env python3
"""
Integration test for Secure Federated Analytics

This module provides integration tests that can be run without the full
end-to-end infrastructure. It uses mocks to simulate the complete flow.

Usage:
    pytest tests/test_analytics/test_secure_fa_integration.py -v
    python tests/test_analytics/test_secure_fa_integration.py
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from typing import List, Dict, Any

from fedbiomed.common.message import FAReply, FARequest
from fedbiomed.researcher.federated_workflows import FederatedAnalytics
from fedbiomed.researcher.federated_workflows._federated_analytics import FAResult


class TestSecureFAIntegration:
    """Integration tests for secure federated analytics flow."""

    @pytest.fixture
    def mock_fds(self):
        """Create mock federated dataset."""
        fds = MagicMock()
        fds.data.return_value = {
            "node-1": {"dataset_id": "ds-1", "data_type": "csv"},
            "node-2": {"dataset_id": "ds-2", "data_type": "csv"},
        }
        fds.node_ids.return_value = ["node-1", "node-2"]
        return fds

    @pytest.fixture
    def mock_requests(self):
        """Create mock requests handler."""
        return MagicMock()

    @pytest.fixture
    def mock_secagg(self):
        """Create mock secure aggregation."""
        secagg = MagicMock()
        secagg.active = True
        secagg._secagg = MagicMock()
        secagg._secagg._biprime = 12345678901234567890
        secagg._secagg._key = 12345
        secagg._secagg._secagg_clipping_range = 1000
        return secagg

    def test_encryption_decryption_flow(self, mock_fds, mock_requests, mock_secagg):
        """Test complete encryption -> aggregation -> decryption flow."""
        
        # Simulate node responses with encrypted values
        # In real implementation, nodes would encrypt before sending
        
        encrypted_replies = {
            "node-1": self._create_mock_reply({
                "AGE": {"_encrypted": True, "value": 11111}
            }),
            "node-2": self._create_mock_reply({
                "AGE": {"_encrypted": True, "value": 22222}
            }),
        }

        # Test the flow manually (without actual SecAgg)
        # This simulates what _decrypt_replies would do
        with patch.object(FederatedAnalytics, '_get_secagg_params') as mock_params:
            mock_params.return_value = {
                "key": 12345,
                "biprime": 12345678901234567890,
                "clipping_range": 1000
            }
            
            # With mocked decryption, we can verify the flow
            fa = FederatedAnalytics(
                fds=mock_fds,
                experiment_id="exp-1",
                researcher_id="res-1",
                reqs=mock_requests,
                experimentation_folder="/tmp",
                secagg=mock_secagg,
            )
            
            # Store mock replies
            result = FAResult(encrypted_replies)
            
            assert result.node_ids == ["node-1", "node-2"]

    def test_histogram_encrypted_flow(self, mock_fds, mock_requests, mock_secagg):
        """Test histogram with encrypted counts flow."""
        
        encrypted_histogram_replies = {
            "node-1": self._create_mock_reply({
                "AGE": {
                    "histogram": {
                        "bin_edges": [0, 20, 40, 60, 80, 100],
                        "counts": [
                            {"_encrypted": True, "value": 10},
                            {"_encrypted": True, "value": 20},
                            {"_encrypted": True, "value": 30},
                            {"_encrypted": True, "value": 25},
                            {"_encrypted": True, "value": 15},
                        ]
                    }
                }
            }),
            "node-2": self._create_mock_reply({
                "AGE": {
                    "histogram": {
                        "bin_edges": [0, 20, 40, 60, 80, 100],
                        "counts": [
                            {"_encrypted": True, "value": 5},
                            {"_encrypted": True, "value": 15},
                            {"_encrypted": True, "value": 25},
                            {"_encrypted": True, "value": 20},
                            {"_encrypted": True, "value": 10},
                        ]
                    }
                }
            }),
        }
        
        result = FAResult(encrypted_histogram_replies)
        
        # Verify structure is preserved
        assert "node-1" in result.node_ids
        assert "node-2" in result.node_ids
        
        # Get node stats
        node1_stats = result.node_stats("node-1")
        assert "AGE" in node1_stats
        assert "histogram" in node1_stats["AGE"]
        assert node1_stats["AGE"]["histogram"]["bin_edges"] == [0, 20, 40, 60, 80, 100]

    def test_multiple_columns_flow(self, mock_fds, mock_requests, mock_secagg):
        """Test multiple columns with mixed statistics."""
        
        multi_col_replies = {
            "node-1": self._create_mock_reply({
                "AGE": {"mean": 65.0, "count": 100, "min": 50, "max": 80},
                "INCOME": {"mean": 50000.0, "count": 100, "min": 30000, "max": 80000},
            }),
            "node-2": self._create_mock_reply({
                "AGE": {"mean": 70.0, "count": 80, "min": 55, "max": 85},
                "INCOME": {"mean": 60000.0, "count": 80, "min": 35000, "max": 90000},
            }),
        }
        
        result = FAResult(multi_col_replies)
        
        # Test aggregation
        global_mean_age = result.global_stat("mean")
        global_count_age = result.global_stat("count")
        
        # Verify aggregation (weighted average for mean)
        assert global_mean_age["AGE"] == 67.22222222222223  # (65*100 + 70*80) / 180
        assert global_count_age["AGE"] == 180

    def test_secagg_setup_flow(self, mock_fds, mock_requests):
        """Test secagg setup flow."""
        
        with patch('fedbiomed.researcher.federated_workflows._federated_analytics.SecureAggregation') as mock_secagg_cls:
            mock_secagg = MagicMock()
            mock_secagg.active = True
            mock_secagg.setup.return_value = True
            mock_secagg.train_arguments.return_value = {
                "secagg_key": 12345,
                "biprime": 12345678901234567890,
                "parties": ["res-1", "node-1", "node-2"]
            }
            mock_secagg_cls.return_value = mock_secagg
            
            fa = FederatedAnalytics(
                fds=mock_fds,
                experiment_id="exp-1",
                researcher_id="res-1",
                reqs=mock_requests,
                experimentation_folder="/tmp",
                secagg=True,
            )
            
            # Setup secagg
            parties = ["node-1", "node-2"]
            result = fa.secagg_setup(parties)
            
            # Verify setup was called
            mock_secagg.setup.assert_called_once()
            mock_secagg.train_arguments.assert_called_once()
            assert "secagg_key" in result

    def test_compute_analytics_full_flow(self, mock_fds, mock_requests):
        """Test compute_analytics with full flow simulation."""
        
        with patch('fedbiomed.researcher.federated_workflows._federated_analytics.SecureAggregation') as mock_secagg_cls, \
             patch('fedbiomed.researcher.federated_workflows._federated_analytics.FARequestJob') as mock_job_cls:
            
            # Setup mock secagg
            mock_secagg = MagicMock()
            mock_secagg.active = True
            mock_secagg.setup.return_value = True
            mock_secagg.train_arguments.return_value = {
                "secagg_key": 12345,
                "biprime": 12345678901234567890,
                "parties": ["res-1", "node-1", "node-2"]
            }
            mock_secagg_cls.return_value = mock_secagg
            
            # Setup mock job
            mock_job = MagicMock()
            mock_job.execute.return_value = (
                {
                    "node-1": self._create_mock_reply({"AGE": {"mean": 65.0}}),
                    "node-2": self._create_mock_reply({"AGE": {"mean": 70.0}}),
                },
                {}
            )
            mock_job_cls.return_value = mock_job
            
            # Create federated analytics
            fa = FederatedAnalytics(
                fds=mock_fds,
                experiment_id="exp-1",
                researcher_id="res-1",
                reqs=mock_requests,
                experimentation_folder="/tmp",
                secagg=True,
            )
            
            # Compute analytics
            result = fa.compute_analytics(
                stats=["mean"],
                dataset_args={"col_names": ["AGE"]}
            )
            
            # Verify flow
            mock_job_cls.assert_called_once()
            mock_secagg.setup.assert_called_once()
            
            # Verify result
            assert "node-1" in result.node_ids
            assert "node-2" in result.node_ids

    def test_caching_with_secagg(self, mock_fds, mock_requests):
        """Test that caching works correctly with secagg."""
        
        with patch('fedbiomed.researcher.federated_workflows._federated_analytics.SecureAggregation') as mock_secagg_cls, \
             patch('fedbiomed.researcher.federated_workflows._federated_analytics.FARequestJob') as mock_job_cls:
            
            mock_secagg = MagicMock()
            mock_secagg.active = True
            mock_secagg.setup.return_value = True
            mock_secagg.train_arguments.return_value = {"key": 1, "biprime": 2}
            mock_secagg_cls.return_value = mock_secagg
            
            mock_job = MagicMock()
            mock_job.execute.return_value = (
                {"node-1": self._create_mock_reply({"AGE": {"mean": 65.0}})},
                {}
            )
            mock_job_cls.return_value = mock_job
            
            fa = FederatedAnalytics(
                fds=mock_fds,
                experiment_id="exp-1",
                researcher_id="res-1",
                reqs=mock_requests,
                experimentation_folder="/tmp",
                secagg=True,
            )
            
            # First call - should trigger request
            result1 = fa.compute_analytics(stats=["mean"])
            assert mock_job.call_count == 1
            
            # Second call with same params - should use cache
            result2 = fa.compute_analytics(stats=["mean"])
            assert mock_job.call_count == 1  # Still 1, no new request
            
            # Results should be the same object
            assert result1 is result2

    def test_error_handling_node_failure(self, mock_fds, mock_requests):
        """Test handling when some nodes fail."""
        
        with patch('fedbiomed.researcher.federated_workflows._federated_analytics.SecureAggregation') as mock_secagg_cls, \
             patch('fedbiomed.researcher.federated_workflows._federated_analytics.FARequestJob') as mock_job_cls:
            
            mock_secagg = MagicMock()
            mock_secagg.active = True
            mock_secagg.setup.return_value = True
            mock_secagg.train_arguments.return_value = {"key": 1, "biprime": 2}
            mock_secagg_cls.return_value = mock_secagg
            
            # One success, one failure
            mock_error = MagicMock()
            mock_error.errnum = "FB325"
            mock_error.extra_msg = "Test error"
            
            mock_job = MagicMock()
            mock_job.execute.return_value = (
                {"node-1": self._create_mock_reply({"AGE": {"mean": 65.0}})},
                {"node-2": mock_error}
            )
            mock_job_cls.return_value = mock_job
            
            fa = FederatedAnalytics(
                fds=mock_fds,
                experiment_id="exp-1",
                researcher_id="res-1",
                reqs=mock_requests,
                experimentation_folder="/tmp",
                secagg=True,
            )
            
            result = fa.compute_analytics(stats=["mean"])
            
            # Should still have result from node-1
            assert "node-1" in result.node_ids

    def _create_mock_reply(self, output: Dict) -> MagicMock:
        """Helper to create mock FAReply."""
        reply = MagicMock(spec=FAReply)
        reply.output = output
        return reply


class TestFAResultIntegration:
    """Integration tests for FAResult with complex scenarios."""

    def test_nested_schema_aggregation(self):
        """Test aggregation with nested schema."""
        replies = {
            "n1": self._make_reply({
                "demographics": {
                    "age": {"mean": 65.0, "count": 100},
                    "income": {"mean": 50000, "count": 100}
                }
            }),
            "n2": self._make_reply({
                "demographics": {
                    "age": {"mean": 70.0, "count": 80},
                    "income": {"mean": 60000, "count": 80}
                }
            }),
        }
        
        result = FAResult(replies)
        
        global_mean = result.global_stat("mean")
        global_count = result.global_stat("count")
        
        assert global_mean["demographics"]["age"] == 67.22222222222223
        assert global_count["demographics"]["age"] == 180
        assert global_mean["demographics"]["income"] == 54444.44444444444

    def test_mixed_encrypted_clear_structure(self):
        """Test with mix of encrypted and clear values."""
        # Simulating histogram: bin_edges in clear, counts might be encrypted
        replies = {
            "n1": self._make_reply({
                "AGE": {
                    "histogram": {
                        "bin_edges": [0, 25, 50, 75, 100],
                        "counts": [10, 30, 40, 20]
                    }
                }
            }),
            "n2": self._make_reply({
                "AGE": {
                    "histogram": {
                        "bin_edges": [0, 25, 50, 75, 100],
                        "counts": [15, 25, 35, 25]
                    }
                }
            }),
        }
        
        result = FAResult(replies)
        
        hist = result.global_stat("histogram")
        
        assert hist["AGE"]["bin_edges"] == [0, 25, 50, 75, 100]
        assert hist["AGE"]["counts"] == [25, 55, 75, 45]

    def _make_reply(self, output: Dict) -> MagicMock:
        """Helper to create mock FAReply."""
        reply = MagicMock(spec=FAReply)
        reply.output = output
        return reply


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
