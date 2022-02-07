# Coding Rules


## Introduction

to be completed

## Code writing rules

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
