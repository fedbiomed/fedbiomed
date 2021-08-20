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

* test coverage

If you want to check the test coverge, you should use:

```
cd tests
nosetests  --cover-html --cover-erase --with-coverage --cover-package=fedbiomed
```

and open the **cover/index.html** file in your favorite browser.
