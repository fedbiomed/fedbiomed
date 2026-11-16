# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Capability guardian example with one node and MNIST (PoC-only).

Runs a short MNIST experiment whose train requests carry a capability, so that
a node with `guardian_service` configured asks the mock inference guardian
whether it may run the training plan. See the "Example with one node" section
of README.md for the node and guardian setup.

Usage (from this folder, with the researcher component):

    python guardian_one_node_example.py --capability valid
    python guardian_one_node_example.py --capability invalid
    python guardian_one_node_example.py --capability none

- `valid`: the capability authorizes the digest of this training plan, rounds run.
- `invalid`: the capability authorizes another digest, the node refuses the round.
- `none`: no capability is sent, the node skips the guardian check.
"""

import argparse
import hashlib
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import transforms

from fedbiomed.common.datamanager import DataManager
from fedbiomed.common.dataset import MnistDataset
from fedbiomed.common.training_plans import TorchTrainingPlan
from fedbiomed.researcher.aggregators.fedavg import FedAverage
from fedbiomed.researcher.federated_workflows import Experiment

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mint_capability import mint_capability  # noqa: E402 (path set just above)


class MnistGuardianPlan(TorchTrainingPlan):
    class Net(nn.Module):
        def __init__(self, model_args):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            return F.log_softmax(self.fc2(x), dim=1)

    def init_model(self, model_args):
        return self.Net(model_args=model_args)

    def init_optimizer(self, optimizer_args):
        return Adam(self.model().parameters(), lr=optimizer_args["lr"])

    def init_dependencies(self):
        return [
            "from torchvision import transforms",
            "from fedbiomed.common.dataset import MnistDataset",
            "from torch.optim import Adam",
        ]

    def training_data(self):
        transform = transforms.Normalize((0.1307,), (0.3081,))
        return DataManager(MnistDataset(transform=transform), shuffle=True)

    def training_step(self, data, target):
        output = self.model().forward(data)
        return torch.nn.functional.nll_loss(output, target)


def main() -> None:
    """Runs the experiment with the requested capability scenario."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--capability", choices=("valid", "invalid", "none"), default="valid"
    )
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()

    exp = Experiment(
        tags=["#MNIST", "#dataset"],
        model_args={},
        training_plan_class=MnistGuardianPlan,
        training_args={
            "loader_args": {"batch_size": 48},
            "optimizer_args": {"lr": 1e-3},
            "epochs": 1,
            "batch_maxnum": 20,
        },
        round_limit=args.rounds,
        aggregator=FedAverage(),
        node_selection_strategy=None,
    )

    # The digest is computed over the exact source sent to the nodes, which
    # includes the dependency imports: take it from the experiment's plan.
    source = exp.training_plan().source()
    digest = "sha256:" + hashlib.sha256(source.encode("utf-8")).hexdigest()
    print(f"Training plan digest: {digest}")

    if args.capability == "valid":
        exp.set_capabilities(mint_capability(digest))
    elif args.capability == "invalid":
        exp.set_capabilities(mint_capability("sha256:" + "0" * 64))

    exp.run()

    print(f"\nCompleted rounds: {sorted(exp.aggregated_params().keys())}")


if __name__ == "__main__":
    main()
