# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""COVID-CT binary classification experiment (PoC-only).

Trains a ResNet-18 to separate COVID from NonCOVID CT slices on the nodes
holding an image-folder dataset tagged `covid-ct` (see README_COVIDCT.md for the
data preparation and node setup). Works with one or several nodes, and can
attach a capability for nodes that have a guardian service configured.

After training, it plots the training loss and the validation loss and
performance (accuracy, sensitivity, specificity, F1) of the global model, and
saves the figure to `--plot-file`.

Usage (from this folder, with the researcher component):

    python covidct_one_node_example.py --rounds 5 --capability none
    python covidct_one_node_example.py --rounds 5 --capability valid

- `valid`: the capability authorizes the digest of this training plan.
- `invalid`: the capability authorizes another digest, guarded nodes refuse.
- `none`: no capability is sent, nodes skip the guardian check.
"""

import argparse
import hashlib
import os
import sys

import torch.nn as nn
from torch.optim import Adam
from torchvision import models, transforms

from fedbiomed.common.datamanager import DataManager
from fedbiomed.common.dataset import ImageFolderDataset
from fedbiomed.common.training_plans import TorchTrainingPlan
from fedbiomed.researcher.aggregators.fedavg import FedAverage
from fedbiomed.researcher.federated_workflows import Experiment

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mint_capability import (  # noqa: E402 (path set just above)
    format_capability,
    mint_capability,
)


class CovidCTClassificationPlan(TorchTrainingPlan):
    def init_model(self, model_args):
        # No pretrained weights: nodes must not download anything
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, model_args["num_classes"])
        return model

    def init_optimizer(self, optimizer_args):
        return Adam(self.model().parameters(), lr=optimizer_args["lr"])

    def init_dependencies(self):
        return [
            "from torch.optim import Adam",
            "from torchvision import models, transforms",
            "from fedbiomed.common.dataset import ImageFolderDataset",
        ]

    def training_data(self):
        # The dataset hands tensors (3 x H x W, values in [0, 1]) to the
        # transform, so only tensor-compatible transforms are used here
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224), antialias=True),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        return DataManager(ImageFolderDataset(transform=transform), shuffle=True)

    def training_step(self, data, target):
        output = self.model().forward(data)
        return nn.functional.cross_entropy(output, target)

    def testing_step(self, data, target):
        # Classes are sorted folder names: CT_COVID = 0 is the positive class
        output = self.model().forward(data)
        pred = output.argmax(dim=1)
        positive, negative = target == 0, target != 0
        tp = (positive & (pred == 0)).sum().item()
        fn = (positive & (pred != 0)).sum().item()
        tn = (negative & (pred != 0)).sum().item()
        fp = (negative & (pred == 0)).sum().item()
        return {
            "loss": nn.functional.cross_entropy(output, target).item(),
            "accuracy": (tp + tn) / max(len(target), 1),
            "sensitivity": tp / max(tp + fn, 1),
            "specificity": tn / max(tn + fp, 1),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        }


# Reference categorical palette (fixed order) and recessive chart ink
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]
SURFACE, TEXT, TEXT_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
PERFORMANCE_METRICS = ["accuracy", "sensitivity", "specificity", "f1"]


def node_labels(exp, node_ids):
    """Builds a display label per node: its name, plus a short id when names repeat.

    Returns:
        labels: dict of `{node_id: label}`
    """
    names = {}
    for replies in (exp.training_replies() or {}).values():
        for node_id, reply in replies.items():
            name = (
                reply.get("node_name")
                if isinstance(reply, dict)
                else getattr(reply, "node_name", None)
            )
            if name:
                names[node_id] = name
    labels = {}
    for node_id in node_ids:
        name = names.get(node_id, "")
        short_id = node_id.removeprefix("NODE_")[:8]
        repeated = list(names.values()).count(name) > 1
        labels[node_id] = (
            f"{name} ({short_id})" if name and repeated else name or short_id
        )
    return labels


def validation_history(store):
    """Extracts validation metrics on global updates, one value per round.

    Validation on global updates at round `r` evaluates the model obtained after
    `r - 1` aggregations, so it is indexed by that number.

    Returns:
        history: dict of `{node_id: {metric: ([aggregations], [values])}}`
    """
    history = {}
    for node_id, phases in store.items():
        for metric, rounds in phases.get("testing_global_updates", {}).items():
            points = sorted((r - 1, v["values"][-1]) for r, v in rounds.items())
            history.setdefault(node_id, {})[metric] = (
                [p[0] for p in points],
                [p[1] for p in points],
            )
    return history


def training_loss_history(store):
    """Extracts the training loss reported by each node.

    Returns:
        history: dict of `{node_id: (batches, means)}` where `batches` is
            `([x], [loss])` with `x = r - 1 + i / n` for the `i`-th of `n`
            reported batches of round `r`, and `means` is `([r - 0.5], [mean loss])`
    """
    history = {}
    for node_id, phases in store.items():
        rounds = phases.get("training", {}).get("Loss", {})
        xs, ys, mean_xs, mean_ys = [], [], [], []
        for r in sorted(rounds):
            iterations, values = rounds[r]["iterations"], rounds[r]["values"]
            n = max(iterations) or 1
            xs += [r - 1 + i / n for i in iterations]
            ys += values
            mean_xs.append(r - 0.5)
            mean_ys.append(sum(values) / len(values))
        history[node_id] = ((xs, ys), (mean_xs, mean_ys))
    return history


def print_results(labels, validation):
    """Prints the validation metrics of the global model as one table per node."""
    header = ["aggregations", "loss"] + PERFORMANCE_METRICS
    for node_id, metrics in validation.items():
        print(f"\nValidation of the global model on {labels[node_id]}")
        print("  " + "".join(f"{h:>13}" for h in header))
        steps = metrics.get("loss", ([], []))[0]
        for k, step in enumerate(steps):
            row = [metrics[m][1][k] for m in header[1:] if m in metrics]
            print(f"  {step:>13}" + "".join(f"{v:>13.3f}" for v in row))


def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=TEXT, fontsize=10.5, fontweight="bold")
    ax.set_xlabel(xlabel, color=TEXT_MUTED)
    ax.set_ylabel(ylabel, color=TEXT_MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT_MUTED, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def plot_history(exp, plot_file, show=False):
    """Plots training loss, validation loss and validation performance.

    The top row compares nodes (one color per node). Below, one panel per node
    shows the performance metrics of the global model (one color per metric).

    Args:
        exp: The experiment, after `run()`.
        plot_file: Path of the PNG file the figure is saved to.
        show: Whether to also open the figure in a window.
    """
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    store = exp.monitor().metric_store()
    validation = validation_history(store)
    training = training_loss_history(store)
    labels = node_labels(exp, store)
    nodes = sorted(store, key=lambda n: labels[n])
    print_results(labels, validation)
    if not nodes:
        print("No training or validation values were received: nothing to plot")
        return

    # Node identity keeps the same color across the top panels. More nodes than
    # palette slots would need small multiples rather than generated hues.
    node_color = {n: SERIES_COLORS[i % len(SERIES_COLORS)] for i, n in enumerate(nodes)}
    node_marker = {n: MARKERS[i % len(MARKERS)] for i, n in enumerate(nodes)}

    n_cols = 2
    perf_keys = [f"P{i}" for i in range(len(nodes))]
    if len(perf_keys) % n_cols:
        perf_keys.append(".")
    mosaic = [["T", "V"]] + [
        perf_keys[i : i + n_cols] for i in range(0, len(perf_keys), n_cols)
    ]
    if len(nodes) == 1:
        mosaic[1] = ["P0", "P0"]
    fig, axes = plt.subplot_mosaic(
        mosaic,
        figsize=(12, 3.6 * len(mosaic) + 0.6),
        facecolor=SURFACE,
        layout="constrained",
    )

    ax = axes["T"]
    for node in nodes:
        (xs, ys), (mean_xs, mean_ys) = training[node]
        ax.plot(xs, ys, color=node_color[node], linewidth=1, alpha=0.3)
        ax.plot(
            mean_xs,
            mean_ys,
            color=node_color[node],
            linewidth=2,
            marker=node_marker[node],
            markersize=7,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            label=labels[node],
        )
    _style(
        ax,
        "Training loss (faint: reported batches, bold: round mean)",
        "Round",
        "Cross-entropy",
    )
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xticks(range(0, len(next(iter(training.values()))[1][0]) + 1))

    ax = axes["V"]
    for node in nodes:
        xs, ys = validation.get(node, {}).get("loss", ([], []))
        ax.plot(
            xs,
            ys,
            color=node_color[node],
            linewidth=2,
            marker=node_marker[node],
            markersize=7,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            label=labels[node],
        )
        if ys:
            ax.annotate(
                f"{ys[-1]:.2f}",
                (xs[-1], ys[-1]),
                xytext=(7, 0),
                textcoords="offset points",
                va="center",
                color=TEXT,
                fontsize=9,
            )
    _style(
        ax,
        "Validation loss of the global model",
        "Aggregations completed",
        "Cross-entropy",
    )
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.margins(x=0.08)
    if len(nodes) > 1:
        fig.legend(
            handles=[
                Line2D(
                    [],
                    [],
                    color=node_color[n],
                    marker=node_marker[n],
                    linewidth=2,
                    markersize=7,
                    markeredgecolor=SURFACE,
                    label=labels[n],
                )
                for n in nodes
            ],
            loc="outside upper right",
            ncols=len(nodes),
            frameon=False,
            labelcolor=TEXT,
            title="Nodes",
            title_fontsize=9,
        )

    for i, node in enumerate(nodes):
        ax = axes[f"P{i}"]
        for metric, color, marker in zip(
            PERFORMANCE_METRICS, SERIES_COLORS, MARKERS, strict=True
        ):
            xs, ys = validation.get(node, {}).get(metric, ([], []))
            ax.plot(
                xs,
                ys,
                color=color,
                linewidth=2,
                marker=marker,
                markersize=7,
                markeredgecolor=SURFACE,
                markeredgewidth=1.5,
                label=metric,
            )
        _style(ax, f"Global model on {labels[node]}", "Aggregations completed", "Score")
        ax.set_ylim(-0.03, 1.03)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                color=c,
                marker=m,
                linewidth=2,
                markersize=7,
                markeredgecolor=SURFACE,
                label=name,
            )
            for name, c, m in zip(
                PERFORMANCE_METRICS, SERIES_COLORS, MARKERS, strict=True
            )
        ],
        loc="outside lower center",
        ncols=len(PERFORMANCE_METRICS),
        frameon=False,
        labelcolor=TEXT,
        title="Validation metrics (COVID = positive class)",
        title_fontsize=9,
    )
    fig.suptitle(
        "COVID-CT federated training",
        x=0.01,
        ha="left",
        color=TEXT,
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(plot_file, dpi=150, facecolor=SURFACE)
    print(f"\nTraining history figure saved to {os.path.abspath(plot_file)}")
    if show:
        plt.show()


def main() -> None:
    """Runs the experiment with the requested capability scenario."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--capability", choices=("valid", "invalid", "none"), default="none"
    )
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--tags", nargs="+", default=["covid-ct"])
    parser.add_argument(
        "--batch-maxnum",
        type=int,
        default=None,
        help="cap on batches per round, to keep a smoke test short",
    )
    parser.add_argument("--plot-file", default="covidct_training_history.png")
    parser.add_argument(
        "--show", action="store_true", help="also open the figure in a window"
    )
    args = parser.parse_args()

    training_args = {
        "loader_args": {"batch_size": 16},
        "optimizer_args": {"lr": 1e-4},
        "epochs": 1,
        "log_interval": 2,
        "test_ratio": 0.15,
        "test_on_global_updates": True,
        "random_seed": 21,
    }
    if args.batch_maxnum is not None:
        training_args["batch_maxnum"] = args.batch_maxnum

    exp = Experiment(
        tags=args.tags,
        model_args={"num_classes": 2},
        training_plan_class=CovidCTClassificationPlan,
        training_args=training_args,
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
    if exp.capabilities():
        print(format_capability(exp.capabilities()))

    exp.run()

    print(f"\nCompleted rounds: {sorted(exp.aggregated_params().keys())}")
    plot_history(exp, args.plot_file, show=args.show)


if __name__ == "__main__":
    main()
