#!/bin/bash -x
# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

bats_file=$(mktemp /tmp/run.bats.XXXXXX)
echo ${bats_file}
exec 3>"$bats_file" # associating a file descriptor with the temp file, so that is removed whatever the reason the script ends.
list_notebooks=( notebooks/101_getting-started.py notebooks/general-breakpoint-save-resume.py notebooks/general-tensorboard.py notebooks/general-use-gpu.py notebooks/pytorch-celeba-dataset.py notebooks/pytorch-csv-data.py notebooks/pytorch-local-training.py notebooks/pytorch-variational-autoencoder.py notebooks/test_nbconvert.py )
#for notebook in ./notebooks/*.py; do
test_counter=1
for notebook in ${list_notebooks[@]:0:3}; do
    echo "adding ${notebook}"
cat <<EOF >>${bats_file}
@test "${test_counter} - $(basename ${notebook})" {
    ./scripts/run_integration_test -s ${notebook}  \
        -d ./tests/datasets/mnist.json \
        -d ./tests/datasets/celeba.json 3>&-
}
EOF
let test_counter+=1
done
#@test "$(basename ${notebook})" {
#    run -0 ./scripts/run_integration_test -s ${notebook}  \
#        -d ./tests/datasets/mnist.json \
#        -d ./tests/datasets/celeba.json >&3
#}

TEST_OUTPUT="integration_tests_outputs${BUILD_NUMBER}"

rm -fr ${TEST_OUTPUT}/*
bats --formatter tap --report-formatter tap --show-output-of-passing-tests -T -x --verbose-run --gather-test-outputs-in ${TEST_OUTPUT} ${bats_file}

# @test "flamby-integration-into-fedbiomed.py" {
#   run ./scripts/run_integration_test -s ./notebooks/flamby-integration-into-fedbiomed.ipynb  \
#                                 -d ./tests/datasets/mnist.json >&2
# }

