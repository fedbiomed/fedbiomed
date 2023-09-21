#!/bin/bash -x
# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

bats_file=$(mktemp /tmp/run.bats.XXXXXX)
echo ${bats_file}
exec 5>"$bats_file" # associating a file descriptor with the temp file, so that is removed whatever the reason the script ends.

rmdir ./data
ln -s ~/Data/fedbiomed ./data

#list_notebooks=( notebooks/101_getting-started.py notebooks/general-breakpoint-save-resume.py notebooks/general-tensorboard.py notebooks/general-use-gpu.py notebooks/pytorch-celeba-dataset.py notebooks/pytorch-csv-data.py notebooks/pytorch-local-training.py notebooks/pytorch-variational-autoencoder.py notebooks/test_nbconvert.py )
list_notebooks=( notebooks/*.ipynb)

test_counter=1
for notebook in ${list_notebooks[@]:0:10}; do # \todo remove the split :..:.. when all notebooks are working.
    echo "adding ${notebook}"
cat <<EOF >>${bats_file}
@test "${test_counter} - $(basename ${notebook})" {
    ./scripts/run_end_to_end_one.sh -s ${notebook}  \
        -d ./tests/datasets/mnist.json  \
        -d ./tests/datasets/test_data_csv.json \
        -d ./tests/datasets/celeba.json 3>&- \
#        3>&-   # indispensable to prevent the tests to hang.
}
EOF
let test_counter+=1
done

TEST_OUTPUT="end_to_end_tests_outputs"
echo "cleaning directory ${TEST_OUTPUT}"
rm -fr ./${TEST_OUTPUT}/*

bats --formatter junit --report-formatter junit --show-output-of-passing-tests -T -x --verbose-run --gather-test-outputs-in ${TEST_OUTPUT} ${bats_file}
