# Coding Rules


## Introduction

to be completed

## Code writing rules

### general

- use getters/setters instead of `@property`

### exceptions handling

- on the node: in general, node should not stop because of exceptions that occur during its operation

- on the researcher: general behaviour is to propagate the exceptions to the top level layer, where they are transformed to a friendlier output. Researcher displays this output and stops.

- when a class raises an exception, it raises a FedbiomedSomethingError: use exceptions defined in **fedbiomed.common.exceptions** :

  Do:
  ```
  raise FedbiomedSometypeError()
  ```

  Don't:
  ```
  raise NameError()
  ```

- optionally, if more specificity is wanted, a class can catch a python (non-Fed-Biomed) exception and re-raise it as a FedbiomedError

  Optionally do:
  ```
  try:
      something()
  except SysError as e:
      raise OneOfFedbiomedError()
  ```

- a class generally shouldn't catch a FedbiomedError and re-raise (a FedbiomedError):

  Don't:
  ```
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

    If needed:
    ```
    try:
      mycode()
    except Exception as e:
      ...
    ```

  - don't use the `except:` clause

    Don't
    ```
    try:
      mycode()
    except:
      ...
    ```

  - should separate FedbiomedError and other exceptions

    Do:
    ```
    try:
      mycode()
    except FedbiomedSomeError as e:
      ...
    except FedbiomedError as e:
      ...
    except Exception as e:
      ...
    ```

    Don't (in case where the Exception can be a FedbiomedError)
    ```
    try:
      some_code()
    except Exception as e:
      ...
    ```

- in general, a class shouldn't log in logger when raising or re-raising an exception. The class should log when catching and not re-raising a FedbiomedError exception. The class can log when catching a and not re-raising a python exception.

  Do:
  ```
  try:
    some_function()
  except FedbiomedSomeError:
    logger.xxx(message)
  ```

  Can do:
  ```
  try:
    some_function()
  except OSError:
    logger.xxx(message)
  ```

  Don't:
  ```
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
  try:
      something()
  except SomeError as e:
      _msg = ErrorNumbers.FBxxx.value + ": the file " + filename + " does not exist"
      raise OneOfFedbiomedError(_msg)
  ```

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
