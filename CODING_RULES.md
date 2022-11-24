# Coding Rules


## Introduction

to be completed

## Code writing rules

### general

- use getters/setters instead of `@property`

### regarding the exceptions


- use exceptions defined in **fedbiomed.common.exceptions**

- callee side: then detecting a python exception in a fedbiomed layer :

  - print a logger.*() message

    - could be logger.critical -> stop the software, cannot continue
      (ex: FedbiomedEnvironError)
    - or logger.error -> software may continue (ex: MessageError)

  - raise the exception as a fedbiomed exception


- caller/intermediate side: trap the exceptions:

  - separate the FedbiomedError and other exceptions

```
try:

    something()

except SysError as e:
    logger.XXX()
    raise OneOfFedbiomedError()
```

  - example of top level function (eg: Experiment())

```
try:

    something()

except FedbiomedError as e:
    etc...

except Exception as e:   <=== objective: minimize the number of type we arrive here !
    # place to do a backtrace
    # extra message to the end user to post this backtrace to the support team
    etc...
```

  - except of the top level program, it is **forbidden** to trap all exceptions (with ```except:``` or ```except Exception```)


- the **try:** block is as small as possible

- force to read the documentation


- string associated to the exception:

  - comes from the fedbiomed.common.constants.ErrorNumbers

  - complemented by a usefull (more precise) information:

  => consider ErrorNumbers as categories

```
try:
    something()

except SomeError as e:

    _msg = ErrorNumbers.FBxxx.value + ": the file " + filename + " does not exist"

    logger.error(_msg)
    raise OneOfFedbiomedError(_msg)
```


- open questions:

  - as a researcher, does I need sometimes the full python backtrace

    - ex: loading a model on the node side

    - ex: debugging fedbiomed itself

    - it should be already a backtrace on the reasearcher/node console

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
