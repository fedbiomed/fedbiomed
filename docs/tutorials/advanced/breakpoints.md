---
title: Breakpoints, Experiment Saving Facility
description: Using breakpoints to save and load each round state during federated training in case of crash or stop. 
keywords: Fed-BioMed experiment, breakpoints, saving breakpoints, loading breakpoints
---

# Breakpoints (experiment saving facility)

An experiment can crash or stop during training due to unexpected events : software component crash (researcher/node), host server failure, network outage, etc. 

If an experiment crashes during training, one can launch an experiment with the same parameters and re-do a similar training. But if the experiment already ran for a long time, we don't want to lose the previous training effort. This is where breakpoints help.

Breakpoints can also be used for a wide variety of reasons such as: the model start to oscillate at some point, the model get over-trained. In such cases, you may want to revert to a previous round's results using a breakpoint, and continue from this breakpoint.

A Fed-BioMed breakpoint is a researcher side function that saves an intermediate status and training results of an experiment to disk files. It can later be used to create an experiment from these saved status and results, to continue and complete the partially run experiment.


## Basic use : continue from last breakpoint of last experiment

!!! note
    Before running the following cells, please make sure you have already a running node. Please follow the following tutorials explaining
    [how to launch Pytorch training plans ](../pytorch/01_PyTorch_MNIST_Single_Node_Tutorial.ipynb)" and
    [Scikit-Learn training plans](../scikit-learn/01_sklearn_MNIST_classification_tutorial.ipynb).
    Don't forget to specify the data under `#dummy_tag` tag.


By default, a Fed-BioMed experiment does not save breakpoints. To create an experiment that save breakpoints after each training round completes, use the `save_breakpoints=True` option :


```python
exp = Experiment(tags=[ '#dummy_tag' ],
                training_plan_class=MyTrainingPlan,
                round_limit=2,
                save_breakpoints=True)
```

or use the `set_breakpoint()` setter
```python
exp.set_save_breakpoints(True)
```

If the experiment crashes or is stopped during training, and at least one round was completed, one can later continue from the last experiment breakpoint.

First step is to stop and restart all Fed-BioMed components still running (researcher, nodes, and sometimes network), as they often have lost their state due to the crash beyond the software automatic recovery capability.

Second step is then to create an experiment on the researcher with status loaded from the last breakpoint of the previously running experiment :

```python
new_exp = Experiment.load_breakpoint()
```

Optionally check the experiment folder and current training round for the loaded breakpoint :

```python
print(f'Experimentation folder {new_exp.experimentation_folder()}')
print(f'Number of experimentation rounds completed {new_exp.round_current()}')
```

Then continue and complete experiment loaded from the breakpoint :

```python
new_exp.run()
```



## Continue from a specific breakpoint and experiment

In some cases, it is needed to indicate which breakpoint should be used to create the experiment, because the last breakpoint of the last experiment cannot be automatically guessed, or because it is desirable to select and continue from an older breakpoint or experiment.

Fed-BioMed saves experiment results and breakpoints with the following file tree structure :
```
<researcher-component-path>/var/experiments
|
\-- Experiment_0
|
\-- Experiment_1
|    |
|    \-- breakpoint_0
|    \-- breakpoint_1
|
\-- Experiment_2
|    |
|    \-- breakpoint_0
|
\-- Experiment_3
```

Result files and breakpoints for each experiment are saved in a distinct folder under `<researcher-component-path>/var/experiments`. By default, each experiment is assigned a folder named `Experiment_{num}` with a unique increasing *num*. When an experiment saves breakpoint, breakpoint for round *round* is saved under `Experiment_{num}/breakpoint_{round}`.

When loading last breakpoint of last experiment, it selects the highest existing *num* experiment and then the highest *round* for this experiment. In this example, automatic selection of last breakpoint fails because last experiment (`Experiment_3`) has not saved breakpoints or could not complete its first round.

To load a specific breakpoint, indicate the path for this breakpoint (absolute or relative to the current directory). For example to load breakpoint after round 1 for `Experiment_1`:

```python
new_exp = Experiment.load_breakpoint(
    "<researcher-component-path>/var/experiments/Experiment_1/breakpoint_1")
```

or, if the current working directory is relative to `<researcher-component-path>` :

```python
# works if current directory is <researcher-component-path>
new_exp = Experiment.load_breakpoint(
    "./var/experiments/Experiment_1/breakpoint_1")
```

Optionally check experimentation loaded from the breakpoint :

```python
print(f'Experimentation path {new_exp.experimentation_path()}')
```


## Limitations


Breakpoints currently do not support copying to another path, as they use absolute path. For example this **doesn't work** in the current version :

```shell
!mv ./var/experiments/Experiment_1 ./var/experiments/mydir
```
```python
new_exp = Experiment.load_breakpoint("./var/experiments/mydir")
```

!!! info
    Breakpoints currently do not save tensorboard monitoring status and tensorboard logs. If you continue from a breakpoint, tensorboard monitoring is not restarted and logs from pre-breakpoint run are not restored.
