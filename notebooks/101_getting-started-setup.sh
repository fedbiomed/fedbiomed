#!/usr/bin/env bash -l
conda activate fedbiomed-node
./scripts/fedbiomed_run node -adff ./tests/datasets/mnist.json

