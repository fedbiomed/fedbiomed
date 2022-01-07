### how to run the tests

* setup the python environment

```
source .../scripts/fedbiomed_environment researcher
```

* run all tests

```
cd tests
python -m unittest -v
```

or
```
cd tests
nosetests -v
```

Because of the code structure (environ singleton), the tests **must** run
in separate processes

* run a specific test

```
cd tests
python ./test_XXX.py
```

or

```
cd tests
nosetests --tests=test_XXX.py
```

Remarks: **nose** could also be used to run the test (same test ficles as with
unittest). One benefit is to have more option to run the test, for example
have a coverage output, xml output for ci, etc...


### doc on unittest

https://docs.python.org/3/library/unittest.html

* test coverage

If you want to check the test coverge, you should use:

```
cd tests
nosetests --cover-xml --cover-erase --with-coverage --cover-package=fedbiomed
coverage html
```

and open the **cover/index.html** file in your favorite browser.

Remark: then using --cover-html instead of --cover-xml, the HTML report does not
contains files which have not been tested, which leads to a over-estimation of
test coverage....


## running an integration test

### global explanation

We provide the script **scripts/run_integration_test** to ease the launching of
tests during the developement process.

The script usage is:

```
Usage: run_integration_test -s file -d dataset.json

  -h, --help                  this help
  -s, --script  <file>        script to run (.py or .ipynb)
  -t, --timeout <integer>     max execution time (default = 900)
  -d, --dataset <json-file>   dataset description

Remark: only dataset availability is checked. Coherence between
provided script and dataset is not validated by this launcher
```

The script algorithm is:
- start the network component
- start one separate node for each provided dataset
- start the researcher component
- stop and clean all components
- the status exit of the script is 0 is everything ran well

The script deals with python scripts or with notebooks.

### dataset description

The datasets are described in a json file, which looks like:

```
{
    "name": "Mednist data",
    "description": "Mednist",
    "tags": "mednist",
    "data_type": "images",
    "path": "$HOME/tmp/MedNIST"
}
```

You can use OS environment variables in this script (e.g. $HOME in the given example)

We provide some example datasets in **tests/datasets**, you may need to adjust them to comply with your own installation directories.


### Examples of use

#### MNIST tutorial

```
$ ./scripts/run_integration_test -s ./notebooks/101_getting-started.py \
                                 -d ./tests/datasets/mnist.json
```

This will run the first tutorial of fed-biomed with one calculation node.


#### monai notebook tutorial with 3 nodes

```
$ ./scripts/run_integration_test \
   -s ./notebooks/monai-2d-image-classification.ipynb \
   -d ./tests/datasets/mednist_part_1.json \
   -d ./tests/datasets/mednist_part_2.json \
   -d ./tests/datasets/mednist_part_3.json \
   -t 3600
```

This will run the monai-2d-image-classification.ipynb notebook, with thres nodes, each of
them using a part of mednist dataset (which has been splitted in three parts).

You may launch this tutorial in a jupyter notebook for more informations.
