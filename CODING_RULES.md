# Coding Rules


## Introduction

This document contains project specific coding rules and guidelines for managing code branches.

## Auto formatter

Fed-BioMed uses the `ruff` tool to enforce flake8 style rules on Python files. Once the development environment is set up by the developer, all necessary tools will be automatically installed. The rules are defined in the `pyproject.toml` file. Developers are free to integrate the Ruff formatter into their IDEs, as long as the IDE uses the configuration specified in `pyproject.toml`. Please refer to `ruff` [documentation](https://docs.astral.sh/ruff/) to see have to use `ruff` for manual execution.

Formatter and linter checks are also applied via a [pre-commit hook](#pre-commit-hook-flake8). Pre-commit tools will be automatically installed once the development environment is correctly set up by the developer.
Please see the [development environment documentation](./docs/developer/development-environment.md) for more details.

### Disable linting for specific lines

In some cases, you may want to intentionally ignore a specific linting rule for a particular line of code. For example, you might have an import that appears unused but is required for side effects, or a long line that you prefer not to break backward compatibility.

To suppress a specific warning or error reported by `ruff`, you can add a `# noqa` comment at the end of the line. This tells the linter to ignore that line during checks. For example:

```python
import numpy as np  # noqa: F401  # disable "imported but unused" warning
```

You can also use just `# noqa` without specifying a rule code to ignore all warnings for that line, but it’s better practice to specify the exact code whenever possible, to avoid hiding unrelated issues.

Use `# noqa` only if there's a valid reason to ignore the linting rule. Overusage of `# noqa` may lead to inconsistent or lower-quality code.

To find the specific rule code reported by `ruff`, you can run `ruff check` locally or examine the pre-commit output, which will indicate the exact rule (e.g., `F401`, `E501`, etc.)




## Code writing rules

### general

- use getters/setters instead of `@property`

### exceptions handling

- on the node: in general, node should not stop because of exceptions that occur while executing requests received from researcher. Top level layer code should catch and handle the exceptions, and can send an error message to the researcher (but without full exception message to avoid leaking information).

- on the researcher: general behaviour is to propagate the exceptions to the top level layer, where they are transformed to a friendlier output. Researcher displays this output and stops.

- when a class raises an exception, it raises a FedbiomedSomethingError, not a python native error: use exceptions defined in **fedbiomed.common.exceptions** or define new ones (eg: one per module) :

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

- keep the **try:** block as small as possible

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
[\`Experiment\`][fedbiomed.researcher.federated_workflows]
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


## Code comments

### FIXME, TODO

Feature branch code comments sometimes contain:

- `FIXME`: potential bug, depending on other feature (so it cannot efficiently be solved now)
- `TODO`: something to do in general

Before attempting to merge into `develop`, if you still have `FIXME` and `TODO` in comments:

- they should be discussed during review
- if kept, an associated issue must be created and referenced (eg `issue #1234`) in the comment


## Branches

Goals for regulating branches usage:

- reach simple and informative history (limit entry number, remove hard to follow graphs)
- keep it simple and safe for developers

These guidelines address the commit/push/pull/merge aspects of branches. Lifecycle guidelines are [available here](./docs/developer/usage_and_tools.md#lifecycle).

### commit messages

As a general rule, for commit messages:

- first line should give big picture of the commit
- optional second line should be blank (to avoid git being confused)
- optional later lines give commit details

Example:
```
Add new function xxx

- implement item1
- improve item2
- remove item3
- fix item4
- item5 not yet working
```

### commit, push, pull

These guidelines mostly apply when working in a feature branch.

#### Commit, Push:
- work freely in your local branch (including using micro-commits, etc.).
- shrink contribution's history to a few significant commits *before pushing* (avoid pushing micro-commits or successive versions of same code to the remote)
  - can use `git commit --amend` to modify your yet-unpushed last commit
  - can squash local commits with `git rebase -i local_branch_base_commit` to shrink the yet-unpushed commits on `local_branch_base_commit`. Typically keep the first commit as `pick`, don't change order of commits, turn the next ones to `squash` to merge them in a single commit.
  - stick to squashing the latest commits (unless you really know what you are doing)

#### Pre-commit hook Flake8

Once changes are committed, `ruff` will run checks to ensure that the modified files comply with the Flake8-style rules defined in `pyproject.toml`. If any lines of code violate these rules, the commit will be blocked, and an error will be displayed.

The recommended practice is to address the issues flagged by `ruff` and then recommit the changes. However, in some cases, the tool may report issues in parts of the code that were not modified in the current commit, or the change requested by the linting tool is not applicable. In such cases, you can bypass the pre-commit hooks by using the `git commit --no-verify` command to skip the linter checks.


#### Pull
- can always use `git pull --rebase` (or add it to configuration to apply by default). No danger, it only rebases local yet-unpushed commits on top of new pulled remote commits.

### update, merge

These guidelines mostly apply when updating a feature branch or merging it to `develop`.

Update a feature branch with latest version of `develop`:
- nice to rebase feature branch on develop (`git rebase develop`) as explained below, rather than merging develop in feature branch (`git merge develop`)
- caveat: don't mix rebase and merging for successive updates of a feature branch!

Merge:
- before merging, if too many commits were pushed to the feature branch, squash *remote branch* with a rebase (see below)

### rebase remote branch

These guidelines mostly apply when updating a feature branch or merging it to `develop`.
- don't rebase `develop` and `master` branches (no rewrite of their history)
- merge the feature branch into `develop` and `master` using a merge commit (default in github).

**Warning !**
- **be careful when using `git rebase` on commits *already pushed to the remote*. Bad use brings a risk of losing or duplicating commits**. Use `git reflog` if you lost some commits.
- use rebase when simple enough, if complicated or doubtful do a merge
- typically stick to squashing the latest commits (unless you really know what you are doing)

How to rebase a feature branch that has already been pushed (examples: rebase on`develop` or squash commits):
- coordinate with other team members
  * OK to rebase when working alone in feature branch
  * when other developers work/commit/push in feature branch, ask if OK to rebase now (eg: Discord), and inform after rebasing is done
- update local clone of `develop` (if rebasing on `develop`) and local clone of `feature_branch` before the rebase
- `git checkout feature_branch`
- if rebasing on `develop` typically use `git rebase -i develop` with default values
- if squashing feature branch typically use `git rebase -i feature_branch_base_commit`, keep the first commit as `pick`, don't change order of commits, turn the commits you want to squash as `squash`.
- manually solve conflicts (if any) to complete rebasing
- push with `push --force-with-lease` (rather than `push -f`) to fail if any conflict with un-coordinated developer (who pushed to feature branch remote in the meantime).
- inform other team members rebase is complete and they need to pull the branch again

