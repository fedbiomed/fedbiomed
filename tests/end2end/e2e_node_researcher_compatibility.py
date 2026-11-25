"""End-to-end node/researcher compatibility using isolated interpreter processes."""

import argparse
import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from importlib.metadata import version
from pathlib import Path

SCRIPT = Path(__file__).resolve()
PLAN_DIR = SCRIPT.parent / "experiments/training_plans"
TAG = "#protocol-compatibility"


def setup_node(root, port):
    from fedbiomed.node.config import node_component
    from fedbiomed.node.dataset_manager import DatasetManager

    print(f"Node: fedbiomed {version('fedbiomed')} ({sys.executable})", flush=True)
    config = node_component.initiate(str(root / "node"))
    config.set("researcher", "ip", "localhost")
    config.set("researcher", "port", str(port))
    config.set("security", "training_plan_approval", "False")
    config.set("authentication", "mutual_authentication", "False")
    config.write()
    csv = root / "training.csv"
    csv.write_text(
        "x1,x2,target\n0,0,0\n0,1,0\n1,0,1\n1,1,1\n2,0,1\n0,2,0\n2,1,1\n0,3,0\n",
        encoding="utf-8",
    )
    dataset_id = DatasetManager(config.getpath("default", "db")).add_database(
        name="Protocol sample",
        data_type="csv",
        tags=[TAG],
        description="Cross-version training",
        path=str(csv),
    )
    (root / "expected.json").write_text(
        json.dumps(
            {
                "node_id": config.get("default", "id"),
                "dataset_id": dataset_id,
            }
        ),
        encoding="utf-8",
    )


def researcher(root, port, node_python):
    # Must precede researcher imports: config and Requests are process singletons.
    print(
        f"Researcher: fedbiomed {version('fedbiomed')} ({sys.executable})", flush=True
    )
    os.environ["FBM_RESEARCHER_COMPONENT_ROOT"] = str(root / "researcher")
    os.environ["FBM_SERVER_HOST"] = "localhost"
    os.environ["FBM_SERVER_PORT"] = str(port)
    from fedbiomed.researcher.aggregators import FedAverage
    from fedbiomed.researcher.config import config
    from fedbiomed.researcher.federated_workflows import Experiment
    from fedbiomed.researcher.requests import Requests

    sys.path.insert(0, str(PLAN_DIR))
    from node_researcher_training_plan import CompatibilityTrainingPlan

    config.set("authentication", "mutual_authentication", "False")
    config.write()
    expected = json.loads((root / "expected.json").read_text(encoding="utf-8"))
    node_id = expected["node_id"]
    requests = Requests(config=config)
    process = None
    try:
        process = subprocess.Popen(
            [
                str(node_python),
                "-c",
                "from fedbiomed.cli import run; run()",
                "node",
                "--path",
                str(root / "node"),
                "start",
            ],
            start_new_session=True,
        )
        # Only startup is retried. Training uses the framework's request policies.
        for _ in range(30):
            assert process.poll() is None, "Node exited during startup"
            if node_id in requests.ping_nodes():
                break
            time.sleep(1)
        else:
            raise AssertionError("Node did not answer ping during startup")
        found = requests.search(tags=[TAG], nodes=[node_id])
        assert set(found) == {node_id}, found
        assert [item["dataset_id"] for item in found[node_id]] == [
            expected["dataset_id"]
        ]
        experiment = Experiment(
            tags=[TAG],
            nodes=[node_id],
            training_plan_class=CompatibilityTrainingPlan,
            model_args={"n_features": 2, "n_classes": 2, "max_iter": 10, "tol": 1e-3},
            training_args={
                "epochs": 1,
                "loader_args": {"batch_size": 1},
                "random_seed": 42,
            },
            round_limit=1,
            aggregator=FedAverage(),
            secagg=False,
        )
        experiment.run()
        assert experiment.round_current() == 1
        replies = experiment.training_replies()[0]
        assert set(replies) == {node_id}, replies
        assert replies[node_id]["success"] is True, replies[node_id]
        assert experiment.aggregated_params()[0]["params"]
        print(
            "PASS: ping, dataset discovery and one training round",
            flush=True,
        )
    finally:
        try:
            if process is not None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                finally:
                    # The CLI parent may exit before its worker children.
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        finally:
            server = requests._grpc_server
            asyncio.run_coroutine_threadsafe(
                server._server.stop(0), server._server._loop
            ).result(10)
            server._thread.join(timeout=10)
            assert not server._thread.is_alive(), "Researcher server did not stop"


def run_pair(node_python, researcher_python):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    with tempfile.TemporaryDirectory(prefix="fbm-protocol-") as tmp:
        with socket.socket() as sock:
            sock.bind(("localhost", 0))
            port = sock.getsockname()[1]
        for python, mode in (
            (node_python, "setup-node"),
            (researcher_python, "researcher"),
        ):
            subprocess.run(
                [
                    str(python),
                    str(SCRIPT),
                    "--mode",
                    mode,
                    "--root",
                    tmp,
                    "--port",
                    str(port),
                    "--node-python",
                    str(node_python),
                ],
                check=True,
                env=env,
                cwd=tmp,
            )


def test_node_researcher_compatibility():
    run_pair(
        Path(os.environ.get("FBM_TEST_NODE_PYTHON", sys.executable)).absolute(),
        Path(os.environ.get("FBM_TEST_RESEARCHER_PYTHON", sys.executable)).absolute(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["run", "setup-node", "researcher"], default="run"
    )
    parser.add_argument("--root", type=Path)
    parser.add_argument("--port", type=int)
    parser.add_argument("--node-python", type=Path)
    parser.add_argument("--researcher-python", type=Path)
    args = parser.parse_args()
    if args.mode == "setup-node":
        setup_node(args.root, args.port)
    elif args.mode == "researcher":
        researcher(args.root, args.port, args.node_python)
    else:
        run_pair(args.node_python, args.researcher_python)


if __name__ == "__main__":
    main()
