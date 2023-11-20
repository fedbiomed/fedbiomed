#!/bin/bash -x
# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

bats_file=$(mktemp /tmp/run.bats.XXXXXX)
echo "${bats_file}"
exec 5>"$bats_file" # associating a file descriptor with the temp file, so that is removed whatever the reason the script ends.

: ${CONDA:=conda}
if [ "$(uname)" == "Darwin" ]; then
  ${CONDA} env update -f ./envs/development/conda/fedbiomed-researcher-macosx.yaml
  ${CONDA} env update -f ./envs/development/conda/fedbiomed-node-macosx.yaml
else
  ${CONDA} env update -f ./envs/development/conda/fedbiomed-researcher.yaml
  ${CONDA} env update -f ./envs/development/conda/fedbiomed-node.yaml
fi
${CONDA} env update -f ./envs/ci/conda/fedbiomed-researcher-end-to-end.yaml

rmdir ./data
ln -s ~/Data/fedbiomed ./data

#list_notebooks=( notebooks/101_getting-started.py notebooks/general-breakpoint-save-resume.py notebooks/general-tensorboard.py notebooks/general-use-gpu.py notebooks/pytorch-celeba-dataset.py notebooks/pytorch-csv-data.py notebooks/pytorch-local-training.py notebooks/pytorch-variational-autoencoder.py notebooks/test_nbconvert.py )

list_notebooks=( notebooks/{\
101_getting-started,\
general-breakpoint-save-resume,\
general-list-datasets-select-node,\
general-tensorboard,\
general-training-plan-approval,\
general-use-gpu,\
}.ipynb)
#list_notebooks=( notebooks/{\
#101_getting-started,\
#general-breakpoint-save-resume,\
#general-list-datasets-select-node,\
#general-tensorboard,\
#general-training-plan-approval,\
#general-use-gpu,\
#pytorch-MNIST-FedProx,\
#pytorch-celeba-dataset,\
#pytorch-csv-data,\
#pytorch-local-training,\
#pytorch-opacus-MNIST,\
#pytorch-variational-autoencoder\
#}.ipynb)

#declearn-with-pytorch,\
#declearn-with-sklearn,\

test_counter=1
for notebook in "${list_notebooks[@]}"; do
    echo "adding ${notebook}"
cat <<EOF >>"${bats_file}"
@test "${test_counter} - $(basename ${notebook})" {
    ./scripts/run_end_to_end_one.sh -s ${notebook}  \
        -d ./tests/datasets/mnist.json  \
        -d ./tests/datasets/test_data_csv.json \
        -d ./tests/datasets/celeba.json 3>&- \
#        3>&-   # indispensable to prevent the tests to hang.
}
EOF
(( test_counter+=1 ))
done

TEST_OUTPUT="end_to_end_tests_outputs"
echo "cleaning directory ${TEST_OUTPUT}"
rm -fr ./${TEST_OUTPUT}/*

bats --formatter junit --report-formatter junit --show-output-of-passing-tests -T -x --verbose-run --gather-test-outputs-in ${TEST_OUTPUT} "${bats_file}"
