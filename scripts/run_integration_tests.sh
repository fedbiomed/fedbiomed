#!/bin/bash -x

bats_file=$(mktemp /tmp/run.bats.XXXXXX)
echo ${bats_file}
exec 3>"$bats_file" # associating a file descriptor with the temp file, so that is removed whatever the reason the script ends.
for notebook in ./notebooks/*101*.py; do
    echo "adding ${notebook}"
cat <<EOF >>${bats_file}
@test "$(basename ${notebook})" {
    run ./scripts/run_integration_test -s ${notebook}  \
        -d ./tests/datasets/mnist.json
}
EOF
done

rm -fr integration_tests_outputs/*
bats --formatter tap --report-formatter tap --show-output-of-passing-tests -T -x --verbose-run --gather-test-outputs-in integration_tests_outputs ${bats_file}

# @test "flamby-integration-into-fedbiomed.py" {
#   run ./scripts/run_integration_test -s ./notebooks/flamby-integration-into-fedbiomed.ipynb  \
#                                 -d ./tests/datasets/mnist.json >&2
# }

