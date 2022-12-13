#!/usr/bin/env -S bats --formatter junit  --report-formatter junit  --verbose-run --show-output-of-passing-tests

@test "notebook 101_getting-started.py" {
  run ./scripts/run_integration_test -s ./notebooks/101_getting-started.py  \
                                 -d ./tests/datasets/mnist.json
}
# @test "flamby-integration-into-fedbiomed.py" {
#   run ./scripts/run_integration_test -s ./notebooks/flamby-integration-into-fedbiomed.ipynb  \
#                                 -d ./tests/datasets/mnist.json >&2
# }

