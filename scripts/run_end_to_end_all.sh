#!/bin/bash -x
# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

bats_file=$(mktemp /tmp/run.bats.XXXXXX)
echo "${bats_file}"
exec 5>"$bats_file" # associating a file descriptor with the temp file, so that is removed whatever the reason the script ends.

: ${CONDA:=conda}

for env_name in researcher node researcher-end-to-end
do
  ${CONDA} env remove --name ${env_name}
done

if [ "$(uname)" == "Darwin" ]; then
  ${CONDA} env update -f ./envs/development/conda/fedbiomed-researcher-macosx.yaml
  ${CONDA} env update -f ./envs/development/conda/fedbiomed-node-macosx.yaml
else
  ${CONDA} env update -f ./envs/development/conda/fedbiomed-researcher.yaml
  ${CONDA} env update -f ./envs/development/conda/fedbiomed-node.yaml
fi
${CONDA} env update -f ./envs/ci/conda/fedbiomed-researcher-end-to-end.yaml
DATA_DIR=./data
if [ -d './fedbiomed/common' ] && [ -d ${DATA_DIR} ] && [ ! -L ${DATA_DIR} ]; then # test if it is the ${DATA_DIR} directory obtained from checkout
  mv -f ${DATA_DIR} ${DATA_DIR}.SAVE
  ln -s ~/Data/fedbiomed ${DATA_DIR}
fi

list_notebooks=( notebooks/{\
101_getting-started,\
general-breakpoint-save-resume,\
general-list-datasets-select-node,\
general-tensorboard,\
general-training-plan-approval,\
general-use-gpu,\
pytorch-MNIST-FedProx,\
pytorch-celeba-dataset,\
pytorch-csv-data,\
pytorch-local-training,\
pytorch-opacus-MNIST,\
pytorch-variational-autoencoder\
}.ipynb)

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
