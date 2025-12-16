from unittest.mock import MagicMock, patch

import pytest

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.dataset_types import DataReturnFormat
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.common.message import ErrorMessage, FAReply, FARequest
from fedbiomed.node.fa_job import FAJob


class TestFAJob:
    @pytest.fixture
    def request_args(self):
        return {
            "researcher_id": "res_1",
            "request_id": "req_1",
            "experiment_id": "exp_1",
            "dataset_id": "dataset_123",
            "fa_id": "fa_1",
            "fa_args": {"mean": ["col1"]},
        }

    @pytest.fixture
    def fa_request(self, request_args):
        return FARequest(**request_args)

    @pytest.fixture
    def fa_job_args(self, fa_request):
        return {
            "root_dir": "/tmp/root",
            "db_path": "/tmp/db.json",
            "node_id": "node_1",
            "node_name": "test_node",
            "request": fa_request,
        }

    @pytest.fixture
    def fa_job(self, fa_job_args):
        return FAJob(**fa_job_args)

    def test_fa_job_init(self, fa_job, fa_job_args, request_args):
        """Test FAJob initialization."""
        assert fa_job._dir == fa_job_args["root_dir"]
        assert fa_job._db_path == fa_job_args["db_path"]
        assert fa_job._node_id == fa_job_args["node_id"]
        assert fa_job._node_name == fa_job_args["node_name"]
        assert fa_job._dataset_id == request_args["dataset_id"]
        assert fa_job._experiment_id == request_args["experiment_id"]
        assert fa_job._fa_id == request_args["fa_id"]
        assert fa_job._researcher_id == request_args["researcher_id"]
        assert fa_job._request_id == request_args["request_id"]
        assert fa_job._fa_args == request_args["fa_args"]

    def test_fa_job_init_defaults(self, fa_job_args, request_args):
        """Test FAJob initialization with default arguments."""
        # Create a request with empty fa_args
        req_args = request_args.copy()
        req_args["fa_args"] = {}
        request = FARequest(**req_args)

        args = fa_job_args.copy()
        args["request"] = request

        job = FAJob(**args)
        assert job._fa_args == {}

    def test_build_error_msg(self, fa_job):
        """Test _build_error_msg method."""
        msg = "Something went wrong"
        errnum = ErrorNumbers.FB313.value
        error = fa_job._build_error_msg(msg, errnum)

        assert isinstance(error, ErrorMessage)
        assert error.extra_msg == msg
        assert error.errnum == errnum
        assert error.node_id == fa_job._node_id
        assert error.request_id == fa_job._request_id

    @patch("fedbiomed.node.fa_job.REGISTRY_CONTROLLERS")
    @patch("fedbiomed.node.fa_job.DatasetManager")
    def test_build_dataset_success(self, mock_dm_cls, mock_registry, fa_job):
        """Test _build_dataset success scenario."""
        mock_dm = mock_dm_cls.return_value
        mock_dm.dataset_table.get_by_id.return_value = {
            "data_type": "csv",
            "path": "/path/to/data",
            "dataset_parameters": {},
        }

        mock_dataset_cls = MagicMock()

        # Setup registry to return our mock dataset class
        # REGISTRY_CONTROLLERS[data_type] returns (loader, saver, dataset_cls)
        mock_registry.__getitem__.return_value = (None, None, mock_dataset_cls)
        mock_registry.__contains__.return_value = True

        dataset = fa_job._build_dataset()

        # Verify dataset creation and initialization
        assert dataset == mock_dataset_cls.return_value
        mock_dataset_cls.return_value.complete_initialization.assert_called_once()
        call_args = mock_dataset_cls.return_value.complete_initialization.call_args
        assert call_args[0][0]["root"] == "/path/to/data"
        assert call_args[0][1] == DataReturnFormat.SKLEARN

    @patch("fedbiomed.node.fa_job.DatasetManager")
    def test_build_dataset_not_found(self, mock_dm_cls, fa_job):
        """Test _build_dataset when dataset is not found in DB."""
        mock_dm = mock_dm_cls.return_value
        mock_dm.dataset_table.get_by_id.return_value = None

        result = fa_job._build_dataset()

        assert isinstance(result, ErrorMessage)
        assert result.errnum == ErrorNumbers.FB313.value
        assert "Did not found proper data" in result.extra_msg

    @patch("fedbiomed.node.fa_job.REGISTRY_CONTROLLERS", {})
    @patch("fedbiomed.node.fa_job.DatasetManager")
    def test_build_dataset_invalid_type(self, mock_dm_cls, fa_job):
        """Test _build_dataset when data type is not supported."""
        mock_dm = mock_dm_cls.return_value
        mock_dm.dataset_table.get_by_id.return_value = {"data_type": "unknown_type"}

        result = fa_job._build_dataset()

        assert isinstance(result, ErrorMessage)
        assert result.errnum == ErrorNumbers.FB313.value
        assert "not supported" in result.extra_msg

    @patch("fedbiomed.node.fa_job.REGISTRY_CONTROLLERS")
    @patch("fedbiomed.node.fa_job.DataLoadingPlan")
    @patch("fedbiomed.node.fa_job.DatasetManager")
    def test_build_dataset_with_dlp_success(
        self, mock_dm_cls, mock_dlp_cls, mock_registry, fa_job
    ):
        """Test _build_dataset with DataLoadingPlan."""
        mock_dm = mock_dm_cls.return_value
        mock_dm.dataset_table.get_by_id.return_value = {
            "data_type": "csv",
            "path": "/path",
            "dlp_id": "dlp_1",
        }
        mock_dm.get_dlp_by_id.return_value = ["dlp_content"]

        mock_dlp = mock_dlp_cls.return_value
        mock_dlp.deserialize.return_value = mock_dlp  # return self

        mock_dataset_cls = MagicMock()
        mock_registry.__getitem__.return_value = (None, None, mock_dataset_cls)
        mock_registry.__contains__.return_value = True

        dataset = fa_job._build_dataset()

        assert dataset == mock_dataset_cls.return_value
        # Check if DLP was passed to complete_initialization
        call_args = mock_dataset_cls.return_value.complete_initialization.call_args
        assert call_args[0][0]["dlp"] == mock_dlp

    @patch("fedbiomed.node.fa_job.REGISTRY_CONTROLLERS")
    @patch("fedbiomed.node.fa_job.DataLoadingPlan")
    @patch("fedbiomed.node.fa_job.DatasetManager")
    def test_build_dataset_dlp_error(
        self, mock_dm_cls, mock_dlp_cls, mock_registry, fa_job
    ):
        """Test _build_dataset when DLP deserialization fails."""
        mock_dm = mock_dm_cls.return_value
        mock_dm.dataset_table.get_by_id.return_value = {
            "data_type": "csv",
            "dlp_id": "dlp_1",
        }
        mock_dm.get_dlp_by_id.return_value = ["dlp_content"]

        mock_dlp_cls.return_value.deserialize.side_effect = FedbiomedError("DLP Error")
        mock_registry.__contains__.return_value = True

        result = fa_job._build_dataset()

        assert isinstance(result, ErrorMessage)
        assert result.errnum == ErrorNumbers.FB313.value
        assert "Cannot recover dlp" in result.extra_msg

    def test_run_success(self, fa_job):
        """Test run method success."""
        mock_dataset = MagicMock()
        mock_dataset.mean.return_value = {"col1": 1.5}

        with patch.object(FAJob, "_build_dataset", return_value=mock_dataset):
            reply = fa_job.run()

            assert isinstance(reply, FAReply)
            assert reply.output["mean"] == {"col1": 1.5}
            assert reply.request_id == fa_job._request_id
            assert reply.node_id == fa_job._node_id

    def test_run_failure(self, fa_job):
        """Test run method when _build_dataset fails."""
        error_msg = ErrorMessage(
            request_id="r",
            researcher_id="res",
            node_id="n",
            node_name="nn",
            extra_msg="err",
            errnum=ErrorNumbers.FB313.value,
        )

        with patch.object(FAJob, "_build_dataset", return_value=error_msg):
            reply = fa_job.run()

            assert isinstance(reply, ErrorMessage)
            assert reply == error_msg
