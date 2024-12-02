# Fed-BioMed Component configuration

**Fed-BioMed components need to be configured before using Fed-BioMed.** The Fed-BioMed CLI simplifies this process by automating the setup of default components to enhance the user experience, especially for testing purposes. This article will guide you through the basic configuration steps required for the minimize initialization of Fed-BioMed components.

For more detailed configuration instructions, please refer to the [Configuring Nodes](../user-guide/nodes/configuring-nodes.md) and [Deployment](../user-guide/deployment/deployment.md) user guides.

---

## Fed-BioMed Components

Fed-BioMed components are instances provided by Fed-BioMed with distinct responsibilities in a federated learning infrastructure. While they can operate independently and do not directly depend on each other, they are necessary to complete a federated learning infrastructure. These components may also include optional sub-components to enhance usability (e.g., the Node GUI, a user-friendly web application for managing nodes).

This guide will be focusing on **Node** and **Researcher** components to create a basic infrastructure to follow the examples and the tutorials in the documentation. To find out more about Fed-BioMed architecture and component please refer to [Fed-BioMed architecture](./fedbiomed-architecture.md)

---

## Initializing Components

**The Fed-BioMed CLI is configured to initialize components if they do not already exist.** Initializing component means creating a specific folder for the component to keep its assets. While Fed-BioMed uses default folder name for each component type it is also possible to use custom paths to address or initialize components with different folder names. For the components that are not existing, the CLI asks for the permission before creating the them.   While this feature allows to create component quickly, CLI also provide a specific option just for initializing/creating [components explicitly](#creating-components-beforehand).

It is important to note that, **unless specified, the component directory will be generated in the directory where the command is executed.** This behavior applies to both the **Researcher** and **Node** components.

!!! note "-y option to create components automatically"
    To avoid approving component creation option `-y` can be used. e.g `fedbiomed node -y [ACTION]` or `fedbiomed researcher -y [ACTION]`. `-y` option should be specified right after component type specification.

### Component directories

Component configurations, and other necessary assets to provide components to function properly are located in its own directory of the component. These directories are called **component directory**, and the directory name (folder name) refers to **component name**.  While Fed-BioMed components come with default components names, it is also possible to use different folder names. This functionality is useful especially if several components are hosted in the same parent directory of the same file system.

First execution of the following command will create will create a component directory called `fbm-node` (default component name) located in the directory where this command is executed, and it will ask for permission to initialize component directory if it is not existing.

```shell
fedbiomed node dataset list
```

Once the default component is created, Fed-BioMed CLI command execution in the same directory will use the default component named `fbm-node`. However, it is also possible to indicate component directory to be able to execute Fed-BioMed command from different directories.

```shell
fedbiomed node --path some/other/directory/fbm-node dataset list
```

This functionality is also same for researcher component.


### Creating components beforehand

Creating components beforehand using the CLI is recommended when creating multiple components before performing any actions on them. A good practice is also to create a separate directory to keep all generated components in one place. For example,  a directory called `fbm-components` can be created to hold all Fed-BioMed components. This will allow to access these component easily.

To create a Node component:

```shell
fedbiomed component create --component node --path fbm-components/my-node
```

To create another Node component:

```shell
fedbiomed component create --component node --path fbm-components/my-second-node
```

The Researcher component is also essential for every Fed-BioMed setup:

```shell
fedbiomed component create --component researcher --path fbm-components/my-researcher
```



## Managing Components


The Fed-BioMed CLI is designed to manage different types of components and their sub-components. It also enables managing multiple components by specifying the path where each component is initialized. This section explains how to address distinct components and minimize errors caused by incorrect component specifications.


### Managing Node Component

All the actions that are specific to Node component should be declared after the option `node` of `fedbiomed` command. For example, `fedbiomed node dataset list` will list all the datasets deployed on the dataset. You can list all possible options and action by executing `fedbiomed node --help`.


The execution of `fedbiomed node` without `--path` option will assume the working directory is the directory where the command is executed, and look for `fbm-node` folder to chose default node instantiation. If this folder is not existing it will ask permission to create one.  Therefore, it is important to double check the directory that `fedbiomed` command is going to me executed.

#### Multi-node setup

As it is mentioned before, `--path` option allows to chose distinct node initialization. This option accepts relative or absolute paths. While managing multiple components it is highly recommended to use `--path` option address correct component initialization.

You can find an example of multi-node initialization;

```
cd my-nodes/
fedbiomed component create --component node --path ./my-node
fedbiomed component create --component node --path ./my-second-node
```
Here is how these components are chosen for specific actions;

```
# List datasets in component initialized in `./my-node`
fedbiomed node --path ./my-node dataset list
```

or,

```
fedbiomed node --path ./my-second-node training-plan list
```

#### Single-node setup

When working with a single-node setup, it is recommended to use default component names to avoid the need for the `--path` declaration each time a `fedbiomed` command is executed.

##### Initializing a default node component

The following command initializes a default Node component:

```shell
fedbiomed component create -c node
```

After initializing a default component, the `fedbiomed node` command can be executed without specifying the `--path` option, **unless it is run from a different directory than where `fbm-node` is located.**

Example:

```shell
# List datasets deployed on the default node
fedbiomed node dataset list
```

##### Initializing a default node component in a specific directory

To initialize a default Node component in a specific directory, component path has to be declared, and the folder should be named `fbm-node`:

```shell
fedbiomed component create -c node -p /path/to/fbm-node
```

If the `fedbiomed` command is executed from a directory other than `/path/to/fbm-node`, the `--path` option must be used to specify the correct Node component:

```shell
# Example with explicit path
fedbiomed node dataset list --path /path/to/fbm-node
```


### Managing researcher component

Unlike the Node component, the **Researcher** serves as the server in the Fed-BioMed federated learning infrastructure. There can only be one active Researcher that Nodes connect to at a time. The default component name for the Researcher is `fbm-researcher`.

While it is possible to create multiple Researcher components, this is only useful when different Researcher configurations are needed to support separate federated learning setups.

#### Creating and Managing Researcher Components

The process of creating and selecting a Researcher component is similar to that of a Node. You can specify a particular Researcher component using the `--path` option.

Example: Default component creation

```shell
fedbiomed component create -c researcher
```

Example: using a researcher component from a specific directory

```shell
fedbiomed researcher --path /path/to/fbm-researcher start
```


#### Choosing the Correct Component Initialization

Unlike Node components, the Researcher component runs within a Python session rather than as a standalone process. This session must be correctly configured to use the intended Researcher component. The `start` command for the Researcher has been designed to ensure to use correct initialization.

Example: Default component creation and starting

```shell
# Default component creation
fedbiomed component create -c researcher

# Start a jupyter notebook for this component
fedbiomed researcher start
```

The commands above will create a default Researcher component (`fbm-researcher`) and start a Jupyter Notebook that is pre-configured to use the created Researcher.

This behavior can also be achieved by specifying the `--path` option when starting the Researcher from a different directory:

```shell
fedbiomed researcher --path /path/to/fbm-researcher start
```


#### Executing Plain Python Scripts Without `fedbiomed researcher`

The Researcher does not need to be started using a Jupyter Notebook. You can use a plain Python script to define and execute experiments. However, in this case, the correct Researcher initialization directory must be set at the beginning of the script.

Example: using environment variables in a script

```python
import os
os.environ["FBM_RESEARCHER_COMPONENT_ROOT"] = '/path/to/fbm-researcher'

# Remaining code for the experiment
# ...
```

Example: Setting environment variables in the command Line

You can achieve the same behavior by exporting the `FBM_RESEARCHER_COMPONENT_ROOT` environment variable before running your script:

```shell
export FBM_RESEARCHER_COMPONENT_ROOT=/path/to/fbm-researcher
python my-experiment.py
```

Alternatively, you can set the environment variable inline:

```shell
FBM_RESEARCHER_COMPONENT_ROOT=/path/to/fbm-researcher python my-experiment.py
```

