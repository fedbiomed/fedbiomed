#!/bin/bash

bats_file=$(mktemp /tmp/run_XXXXXX.bats)
exec 3>"$bats_file" # file descriptor 3 
for notebook in ./notebooks/*.py; do
    echo "adding ${notebook}"
cat <<EOF >>${bats_file}
@test "${notebook}" {
    run ./scripts/run_integration_test -s ${notebook}  \
        -d ./tests/datasets/mnist.json
}
EOF
done
echo "---- content of bats_file, begin -------"
cat ${bats_file}
echo "---- content of bats_file, end -------"
bats --formatter junit  --report-formatter junit  --verbose-run --show-output-of-passing-tests ${bats_file} 

# @test "flamby-integration-into-fedbiomed.py" {
#   run ./scripts/run_integration_test -s ./notebooks/flamby-integration-into-fedbiomed.ipynb  \
#                                 -d ./tests/datasets/mnist.json >&2
# }

