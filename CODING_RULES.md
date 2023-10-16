# Coding Rules


## Introduction

to be completed

## Code writing rules

### general

- use getters/setters instead of `@property`

### exceptions handling

- on the node: in general, node should not stop because of exceptions that occur while executing requests received from researcher. Top level layer code should catch and handle the exceptions, and can send an error message to the researcher (but without full exception message to avoid leaking information).

- on the researcher: general behaviour is to propagate the exceptions to the top level layer, where they are transformed to a friendlier output. Researcher displays this output and stops.

- when a class raises an exception, it raises a FedbiomedSomethingError: use exceptions defined in **fedbiomed.common.exceptions** :

  Do:
  ```
  from fedbiomed.common.exceptions import FedbiomedSometypeError
  raise FedbiomedSometypeError()
  ```

  Don't:
  ```
  raise NameError()
  ```

- optionally, if more specificity is wanted, a class can catch a python (non-Fed-Biomed) exception and re-raise it as a FedbiomedError

  Optionally do:
  ```
  from fedbiomed.common.exceptions import OneOfFedbiomedError
  try:
      something()
  except SysError as e:
      raise OneOfFedbiomedError()
  ```

- a class generally shouldn't catch a FedbiomedError and re-raise (a FedbiomedError):

  Don't:
  ```
  from fedbiomed.common.exceptions import FedbiomedSometypeError, FedbiomedOnetypeError
  try:
    somecode()
  except FedbiomedSometypeError;
    raise FedbiomedOnetypeError()
  ```

- when catching exceptions

  - try to be specific about the exception if easy/meaningful:

    Ideally:
    ```
    try:
      mycode()
    except ValueError as e:
      ...
    ```

  - can use the `except Exception:` when re-raising (usually in lower layers, for error message specificity)

    If needed:
    ```
    from fedbiomed.common.exceptions import FedbiomedSomeError
    try:
      mycode()
    except Exception as e:
      raise FedbiomedSomeerror
    ```

  - should use the `except Exception:` in top layer code for handling unexpected errors. On the node, exception is not re-raised and an error message is sent to the researcher.

  - don't use the very general `except:` clause

    Don't
    ```
    try:
      mycode()
    except:
      ...
    ```

  - can separate FedbiomedError and other exceptions when possible/meaningful to take distinct actions

    Can do:
    ```
    from fedbiomed.common.exceptions import FedbiomedSomeError, FedbiomedError
    try:
      mycode()
    except FedbiomedSomeError as e:
      ...
    except FedbiomedError as e:
      ...
    except Exception as e:
      ...
    ```

- in general, a class shouldn't log in logger when raising or re-raising an exception. The class should log when catching and not re-raising a FedbiomedError exception. The class can log when catching a and not re-raising a python exception.

  Do:
  ```
  from fedbiomed.common.logger import logger
  try:
    some_function()
  except FedbiomedSomeError:
    logger.xxx(message)
  ```

  Can do:
  ```
  from fedbiomed.common.logger import logger
  try:
    some_function()
  except OSError:
    logger.xxx(message)
  ```

  Don't:
  ```
  from fedbiomed.common.exceptions import FedbiomedSomeException
  from fedbiomed.common.logger import logger
  try:
    some_function()
  except AnException:
    logger.xxx(message)
    raise FedbiomedSomeException()
  ```

- keep the **try:** block is as small as possible

- string associated to the exception:

  - comes from the fedbiomed.common.constants.ErrorNumbers

  - can be complemented by a useful (more precise) information:

  => consider ErrorNumbers as categories

  ```
  from fedbiomed.common.exceptions import OneOfFedbiomedError
  from fedbiomed.common.constants import ErrorNumbers
  try:
      something()
  except SomeError as e:
      _msg = ErrorNumbers.FBxxx.value + ": the file " + filename + " does not exist"
      raise OneOfFedbiomedError(_msg)
  ```

### arguments checking

Arguments checking means verifying functions' argument types and values.

- in general, do argument checking when either:
  - methods are exposed to an external input (eg: user input, receive data from network, import a file)
  - arguments are security significant (need to check to ensure some security condition)

- in general, don't do argument checking:
  - most of the time for application internal interfaces
  - in particular: no argument check for private methods

- we might want to add argument checking at a few carefully chosen functional boundaries, for modularity/robustness sake

- argument checking should be done as near as possible to the acquisition of the data

## Docstrings writing rules

- use [google style docstrings](https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings)
- optional extensions you can use:
  - info captions
```bash
!!! info "Some info title"

  Some more text for the caption with indent

Not part of the caption anymore etc.
```
  - links to classes in docstrings
```bash
[\`Experiment\`][fedbiomed.researcher.experiment]
```


- if method returns `tuple` or `list` of different types, please follow the example below

```python
    def _configure_multiclass_parameters(y_true: np.ndarray,
                                         y_pred: np.ndarray,
                                         parameters: Dict[str, Any],
                                         metric: str) -> Tuple[np.ndarray, np.ndarray, str, int]:
        """Re-format data giving the size of y_true and y_pred,

        In order to compute classification testing metric. If multiclass dataset, returns one hot encoding dataset.
        else returns binary class dataset

        Args:
            y_true: labels of data needed for classification
            y_pred: predicted values from the model. y_pred should tend towards y_true, for an optimal classification.
            parameters: parameters for the metric (metric_testing_args).

                Nota: If entry 'average' is missing, defaults to 'weighted' if multiclass dataset, defaults to 'binary'
                otherwise. If entry 'pos_label' is missing, defaults to 1.
            metric: name of the Metric

        Returns:
            As `y_true`, reformatted y_true
            As `y_pred`, reformatted y_pred
            As `average`, method name to be used in `average` argument (in the sklearn metric)
            As `pos_label`, label in dataset chosen to be considered as the positive dataset. Needed
                for computing Precision, recall, ...
        """
```
