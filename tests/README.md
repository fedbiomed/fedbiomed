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

* run a specific test

```
cd tests
python ./test_XXX.py
```

Remark: **nose** could also be used to run the test (same test ficles as with
unittest). One benefit is to have more option to run the test, for example
have a coverage output, xml output for ci, etc...

### doc on unittest

https://docs.python.org/3/library/unittest.html
