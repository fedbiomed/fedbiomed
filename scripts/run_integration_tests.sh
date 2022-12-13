#!/bin/bash

bats_file=$(mktemp /tmp/run.bats.XXXXXX)
echo ${bats_file}
exec 3>"$bats_file" # file descriptor 3 
for notebook in ./notebooks/*101*.py; do
    echo "adding ${notebook}"
cat <<EOF >>${bats_file}
@test "${notebook}" {
    run ./scripts/run_integration_test -s ${notebook}  \
        -d ./tests/datasets/mnist.json
}
EOF
done

bats --formatter tap --report-formatter tap --verbose-run --show-output-of-passing-tests ${bats_file} 

# @test "flamby-integration-into-fedbiomed.py" {
#   run ./scripts/run_integration_test -s ./notebooks/flamby-integration-into-fedbiomed.ipynb  \
#                                 -d ./tests/datasets/mnist.json >&2
# }

