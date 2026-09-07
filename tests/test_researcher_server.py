import os
import stat
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from fedbiomed.common.config import Config
from fedbiomed.common.exceptions import (
    FedbiomedCommunicationError,
    FedbiomedResearcherServerError,
)
from fedbiomed.common.utils import create_fedbiomed_setup_folders
from fedbiomed.researcher.config import ResearcherConfig
from fedbiomed.researcher.server import (
    ResearcherServer,
    create_app,
    ensure_token,
    token_path,
)

DATASETS = {"NODE_1": [{"name": "MNIST", "tags": ["#MNIST"]}]}


@pytest.fixture(name="researcher_config", scope="module")
def researcher_config_fixture(tmp_path_factory):
    """A researcher component of its own, so these tests never read the global one.

    `Config.vars` is shared by every configuration object, so it is put back as it
    was: the rest of the suite drives the global component through it.
    """
    saved_vars = dict(Config.vars)
    root = str(tmp_path_factory.mktemp("fbm-researcher-api"))
    create_fedbiomed_setup_folders(root)

    yield ResearcherConfig(root=root)

    Config.vars.clear()
    Config.vars.update(saved_vars)


@pytest.fixture(name="requests")
def requests_fixture() -> MagicMock:
    """A double of the request handler, so no gRPC server is started"""
    requests = MagicMock()
    requests.is_ready.return_value = True
    requests.get_nodes.return_value = [
        {"node_id": "NODE_1", "status": "ACTIVE"},
        {"node_id": "NODE_2", "status": "DISCONNECTED"},
    ]
    requests.ping_nodes.return_value = ["NODE_1"]
    requests.list.return_value = DATASETS
    requests.search.return_value = DATASETS

    return requests


@pytest.fixture(name="stopped_server")
def stopped_server_fixture(researcher_config) -> ResearcherServer:
    """A server whose transport layer never comes up"""
    server = ResearcherServer(config=researcher_config)
    server.start_transport = lambda: None

    return server


@pytest.fixture(name="server")
def server_fixture(researcher_config, requests) -> ResearcherServer:
    """A server whose transport layer is replaced by a double"""
    server = ResearcherServer(config=researcher_config)
    server.start_transport = lambda: setattr(server, "_requests", requests)

    return server


@pytest.fixture(name="anon_client")
def anon_client_fixture(server):
    """A client carrying no credentials"""
    with TestClient(create_app(server), raise_server_exceptions=False) as client:
        yield client


@pytest.fixture(name="client")
def client_fixture(anon_client, server):
    """A client carrying the API token of the server"""
    anon_client.headers.update({"Authorization": f"Bearer {server.get_api_token()}"})

    return anon_client


def test_api_token_01_is_readable_by_its_owner_only(researcher_config):
    path = token_path(researcher_config)
    if os.path.isfile(path):
        os.remove(path)

    token = ensure_token(researcher_config)

    assert token
    assert stat.S_IMODE(os.stat(path).st_mode) == stat.S_IRUSR | stat.S_IWUSR


def test_api_token_02_is_stable_across_restarts(researcher_config):
    assert ensure_token(researcher_config) == ensure_token(researcher_config)


def test_api_token_03_unwritable_location_is_reported(researcher_config, tmp_path):
    unreachable = str(tmp_path / "no-such-directory" / "api_token")

    with patch(
        "fedbiomed.researcher.server._auth.token_path", return_value=unreachable
    ):
        with pytest.raises(FedbiomedResearcherServerError, match="FB634"):
            ensure_token(researcher_config)


def test_server_01_binds_to_configured_api_address(researcher_config):
    server = ResearcherServer(config=researcher_config)

    assert server.get_host() == researcher_config.get("api", "host")
    assert server.get_port() == int(researcher_config.get("api", "port"))


def test_server_02_arguments_override_configuration(researcher_config):
    server = ResearcherServer(config=researcher_config, host="0.0.0.0", port=9999)

    assert server.get_host() == "0.0.0.0"
    assert server.get_port() == 9999


def test_server_03_requests_refused_before_transport_starts(researcher_config):
    server = ResearcherServer(config=researcher_config)

    with pytest.raises(FedbiomedResearcherServerError, match="FB634"):
        server.get_requests()

    assert server.is_transport_ready() is False


def test_server_04_transport_starts_once(researcher_config):
    server = ResearcherServer(config=researcher_config)

    with patch(
        "fedbiomed.researcher.server._server.Requests", autospec=True
    ) as requests_cls:
        requests_cls.return_value.is_ready.return_value = True

        server.start_transport()
        server.start_transport()

        requests_cls.assert_called_once_with(config=researcher_config)

    assert server.get_requests() is requests_cls.return_value
    assert server.is_transport_ready() is True


def test_health_01_is_open_and_reports_transport(anon_client, researcher_config):
    response = anon_client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["researcher_id"] == researcher_config.get("default", "id")
    assert body["transport"] == {
        "host": researcher_config.get("server", "host"),
        "port": int(researcher_config.get("server", "port")),
        "ready": True,
    }


def test_health_02_reports_a_transport_that_did_not_start(stopped_server):
    with TestClient(create_app(stopped_server)) as client:
        assert client.get("/health").json()["transport"]["ready"] is False


def test_nodes_01_lists_known_nodes(client):
    response = client.get("/api/v1/nodes")

    assert response.status_code == 200
    assert response.json() == [
        {"node_id": "NODE_1", "status": "ACTIVE"},
        {"node_id": "NODE_2", "status": "DISCONNECTED"},
    ]


def test_nodes_02_ping_reports_answering_nodes(client, requests):
    response = client.post("/api/v1/nodes/ping")

    assert response.status_code == 200
    assert response.json() == ["NODE_1"]
    requests.ping_nodes.assert_called_once()


def test_datasets_01_lists_every_dataset(client, requests):
    response = client.get("/api/v1/datasets")

    assert response.status_code == 200
    assert response.json() == DATASETS
    requests.list.assert_called_once_with(nodes=None)
    requests.search.assert_not_called()


def test_datasets_02_listing_filters_the_questioned_nodes(client, requests):
    client.get("/api/v1/datasets", params={"nodes": ["NODE_1"]})

    requests.list.assert_called_once_with(nodes=["NODE_1"])


def test_search_01_looks_for_the_given_tags(client, requests):
    response = client.post("/api/v1/datasets/search", json={"tags": ["#MNIST"]})

    assert response.status_code == 200
    assert response.json() == DATASETS
    requests.search.assert_called_once_with(tags=["#MNIST"], nodes=None)
    requests.list.assert_not_called()


def test_search_02_filters_the_questioned_nodes(client, requests):
    client.post(
        "/api/v1/datasets/search", json={"tags": ["#MNIST"], "nodes": ["NODE_1"]}
    )

    requests.search.assert_called_once_with(tags=["#MNIST"], nodes=["NODE_1"])


def test_search_03_accepts_a_tagless_query(client, requests):
    """A tag-less search matches every dataset, as `set([]).issubset(...)` does"""
    response = client.post("/api/v1/datasets/search", json={})

    assert response.status_code == 200
    requests.search.assert_called_once_with(tags=[], nodes=None)


def test_search_04_refuses_a_malformed_query(client, requests):
    response = client.post("/api/v1/datasets/search", json={"tags": "#MNIST"})

    assert response.status_code == 422
    requests.search.assert_not_called()


@pytest.mark.parametrize(
    "headers",
    [{}, {"Authorization": "Bearer wrong-token"}, {"Authorization": "Basic x"}],
    ids=["no-credentials", "wrong-token", "wrong-scheme"],
)
def test_auth_01_refuses_requests_without_the_token(anon_client, headers):
    response = anon_client.get("/api/v1/nodes", headers=headers)

    assert response.status_code == 401


def test_errors_01_unreachable_transport_is_reported_as_unavailable(client, requests):
    requests.get_nodes.side_effect = FedbiomedCommunicationError("FB628: no server")

    response = client.get("/api/v1/nodes")

    assert response.status_code == 503
    assert "FB628" in response.json()["detail"]


def test_errors_02_transport_not_started_is_reported_as_unavailable(stopped_server):
    with TestClient(
        create_app(stopped_server), raise_server_exceptions=False
    ) as client:
        response = client.get(
            "/api/v1/nodes",
            headers={"Authorization": f"Bearer {stopped_server.get_api_token()}"},
        )

    assert response.status_code == 503
    assert "FB634" in response.json()["detail"]
