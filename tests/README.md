## unit tests

### material

tests are run with [nosetests](https://nose.readthedocs.io/en/latest/testing.html), which uses unittests as test framework.
nosetests provide additional features:
- more assert capabilities
- coverage report, which provides output integrated on our Continous Integration platform


### how to run the tests

* setup the python environment

```
source ../scripts/fedbiomed_environment researcher
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

* run all tests from a specific file

```
cd tests
python ./test_XXX.py
```

or

```
cd tests
nosetests --tests=test_XXX.py
```

* run a specific single test. You must specify all the path to this specific test (test\_file.py:TesctClass.specific\_test\_to_run). Eg:

```
cd tests
nosetests test_message.py:TestMessage.test_dummy_message
```

Remarks: **nose** could also be used to run the test (same test files as with
unittest). One benefit is to have more option to run the test, for example
have a coverage output, xml output for ci, etc...

Remarks: coverage is configured via the **.coveragerc** file situated at top directory. Documentation available here:
https://coverage.readthedocs.io/en/stable/config.html

### How to write Unit Tests with `unittest` framework: coding conventions

Mocks are objects that isolate the behaviour of an existing class and simulate it by an object less complex. Better said, Mocking is creating objects that simulate the behavior of real objects

* **Patchers**: patchers are Mocking objects, that replace the call to a method
by a specified value / method. Usually, we want that this method to be very simple in comparison to the patched method

  - how to use patchers?
    - overriding call to class constructors
    - overriding call to class method
    - overriding call to builtins functions
  - `return_value` attribute
  - `side_effect` attribute

* **MagicMock**: a Mocking object that is behaving like a fake class. One can
add extra methods to this object
  - `return_value` attribute
  - `side_effect` attribute
  - `spec` attribute

* **Tests**:
  - testing returned parameters / modified class attributes
  - testing the call of specific methods (with correct values)
  - testing the raise of Exceptions
  - testing the log generation

### doc on unittest

https://docs.python.org/3/library/unittest.html

* test coverage

If you want to check the test coverage, you should use:

```
cd tests
nosetests --cover-xml --cover-erase --with-coverage --cover-package=fedbiomed
coverage html
```

and open the **htmlcov/index.html** file in your favorite browser.

Remark: then using --cover-html instead of --cover-xml, the HTML report does not
contains files which have not been tested, which leads to a over-estimation of
test coverage....


## how to write a new test

### test filename

The filename of a test should start with **test_** and be connected to the tested class or tested file (eg: **test\_exceptions.py**)

### basic skeleton

The basic structure of a test file is:

```
import unittest

from   fedbiomed.YYY import XXX

class TestXXX(unittest.TestCase):
    '''
    Test the XXX class
    '''
    def setUp(self):
        '''
        before each individual test
        '''
        pass

    def tearDown(self):
        '''
        after each individual test
        '''
        pass


    def test_XXX_01_whatever(self):
        '''
        test the whatever feature
        '''

        # some code
        ...
        self.assertEqual( data, expected)
        pass


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
```

Remark: a test file may implement more than one class (we may for example create a **TestMessagingResearcher** and a **TestMessagingNode** in the same **test\_messaging.py** file.

We may also provide two test file for the same purpose. The choice depend on the content of the tests and on the code which can be shared between these files (for example, we may implement a connexion to a MQTT server in a setupMethod() if we include the two classes in a single test file.

### How to mock environ in a test

Environ for the components are mocked by the classes `NodeTestCase` and `ResearcherTestCase`. 
Please use `NodeTestCase` for the classes that uses `environ` on the node side adn `ResearcherTestCase`
for the classes on researcher side. Your test class should inherit one of them. Example:

```python

class TestDatasetManager(NodeTestCase):
  
    #......
    
    
if __name__ == '__main__':
    unittest.main()
```

NodeTestCase or ResearcherTestCase should be imported before importing any other Fed-BioMed or test support modules.

**Note:** `cls.env` can be used in order to modify/remove environ variable values.

**IMPORTANT**: It is required to add `super().setUpClass()` in child's `setUpClass` method if it is added into test 
class. Same for `tearDownClass()`



#### test\_environ\_common.py

This file tests the fedbiomed.common.environ module. It is the only test which does not use fake\_*\_environ.
Reciprocally: all others tests **must** use the mocked environ.


### setUp/tearDown

These methods are called before and after each test block. It is not mandatory to use it, it only eases the definition of the starting conditions of a test

### setUpClass/tearDownClass

These **@classmethod** methods are called **once** at the beginning (just after) and end (just before) of the class creation.

These method are class methods (the decorator is mandatory), and takes **cls** (and not **self**) as argument.

These classes are usually used to:
- install mocking/patch mechanism if several tests use the same mock or patch.
- concentrate all costly initialization (huge file reading, initialize a connexion with an external server, etc..) . Be careful that the test do not break this initialization.

More info: https://docs.python.org/3/library/unittest.html#class-and-module-fixtures


### setUpModule/teardownModule

Same concept but at the file level.


### test naming

Inside a test file (**test_XXX.py**), we propose to number the tests as follow:

```
def test_XXX_01_description_for_this_testCase(self):
   ...


def test_XXX_02_other_description(self):
   ...
```

### input

If a test relies on an input file, the input file is stored in the **test-data** subdiretory and is part of the git repository.

Be carefull with the input file size, specially in case of binary files (images,...) since git has trouble handling them.


### output

Temporary output must be stored in temporary and random directories and destroyed at the end of the test.

You can use the python **tempfile** package for this purpose.


### mock / patch

To test some of our module, we need to simulate input/ouput of other fedbiomed modules.

#### mock / patch location

The mock/patch code may be mutualized between tests, we propose to store this code in the **testsupport** directory and name the files as **fake\_something.py**


#### example on how to patch XXX

...to be continued...

### some usefull tricks

#### provide a \_\_main\_\_ entry to a test file

If you terminate each test file with the two following lines:

```
if __name__ == '__main__':  # pragma: no cover
    unittest.main()
```

you can run the test with the simple ```python ./test_file.py``` command.


#### report all files in code coverage

The **test\_insert\_untested\_python\_files\_here.py** file contains all files of the fedbiomed package.
Its purpose is to provide a code coverage report for all files of fedbiomed library, even if not proper
unit test is provided for the file.

Of course, this is a temporary situation, waiting for all files to be tested properly.


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

This will run the first tutorial of Fed-BioMed with one calculation node.


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

#### sklear perceptron tutorial with 3 nodes

First, create the c1.csv. c2.csv, c3.csv files as described in the notebook, then:

```
../scripts/run_integration_test \
  -s ../notebooks/sklearn-perceptron.ipynb \
  -d ./datasets/sklearn_perceptron_1.json \
  -d datasets/sklearn_perceptron_2.json \
  -d datasets/sklearn_perceptron_3.json
```
