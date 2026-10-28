# Security: Training Plan Security Manager

Federated learning in Fed-BioMed is performed by sending training plans to each node and by requesting nodes to train model from the training plan. Hence, the content of training plan files plays an important role in terms of privacy and security. Therefore, Fed-BioMed offers a feature to run only pre-approved training plans. Training plan files that are sent by a researcher with the training request must be previously approved by the node side. This training plan approval process avoids possible attacks that come through the training plan code files. By running into nodes, they may access private data or jeopardize the node. The approval process should be done by a real user/person who reviews the training plan file. The reviewer should make sure the training plan doesn't contain any code that might cause privacy issues or harm the node.

We talk about "Training plan registration" as it is a whole training plan that is registered, approved and controlled on a node, not only the training plan's model.


## How the Training Plans get Registered, Approved and Controlled

Training Plan registration and approval can be done through the Fed-BioMed node CLI (Command Line Interface) ou GUI (Graphical User Interface) before or after the node is started. Each training plan should have a unique name, a unique file path and a unique hash.

- During the **registration**, the `TrainingPlanSecurityManager` of the Fed-BioMed hashes the content of the training plan file and saves the hash into the persistent database.
- During the **approval**, the training plan is then marked with `Approved` status in the database (`Approved Training Plans` in Figure 1).
- Then when training begins, training plan files that are sent by a researcher are first hashed and compared to the saved hashes of approved training plans: this is training plan **control**. This process flow is presented in the following figure.


![training-plan-approval-process](../../assets/img/diagrams/model-approval-process_v3_training_plan.jpg#img-centered-lr)
*Figure 1 - Controlling training plan file requested by researcher*

Details of `Figure 1` workflow:


1. `Researcher` creates a `TrainingPlan` to be trained on the `Node`
2. `Researcher` submits a `TrainingPlan` along with the training request
3. `Node` retrieves `TrainingPlan` class.
4. `Node` computes hash of the incoming `TrainingPlan`.
5. `Node` checks if `Researcher`'s training plan has already been approved by  comparing its hash to the existing pool of approved training plan hashes.
6. If `TrainingPlan` has been approved by the `Node`, `Researcher` will be able to train his/her training plan on the `Node`.


### Hashing

`TrainingPlanSecurityManager` hashes the minified source with SHA256. The algorithm is fixed for both registration and verification; it is not a node setting.

### Checksum Operation

Checksum control operation is done by querying database for hash of the requested training plan. Therefore, the training plan files that are used should be saved and approved into database by using Fed-BioMed CLI ou GUI before the checksum verification operation (train request). This operation controls whether any existing and approved training plan matches the training plan requesting to train on the node.

`TrainingPlanSecurityManager` minifies training plan files just before hashing the training plan file. The minification process removes spaces and comments from the training plan file. The purpose of using minified training plans is to avoid errors when the requested training plan file has the same code but more or less comments or empty spaces than the training plan which is approved. Since the spaces and the comments will have no effect when executing training plans, this process will not open a back door for attacks. Therefore, having more spaces or comments than the registered training plan will not affect the checksum result (and thus the hashing).


## Managing Nodes for Training Plan Control

Training Plan Control activation on nodes can be managed either through configuration file or Fed-BioMed CLI. The `security` section of the node configuration enables or disables training plan control. By default, Fed-BioMed does not enable training plan control. It means when you start or add data to the node for the first time, if the configuration file doesn't exist, it creates a new configuration file with training plan control disabled (`training_plan_approval = False`).

### Tutorial training plans

Tutorial plans follow the same review process as other plans. When approval mode is enabled, submit the plan from the researcher or register a reviewed file locally, then approve it on each node. Starting a node never registers or approves tutorial code.

### Config Files

When the new node is created without any specified configuration file or any options, the default configuration file is saved into the `etc` directory of Fed-BioMed as follows.

```buildoutcfg
[default]
# other parameters

[researcher]
# parameters connecting researcher server

[security]
training_plan_approval = False
# other security parameters

[researcher]
# parameters for grpc

# etc.
```
As you can see, by default, training plan control (`training_plan_approval`) is disabled. For enabling or disabling this feature, you can change its value to `True` or `False`. Any values different from `True` or `False` will be counted as `False`. The node should be restarted to apply changes after updating the config file.

### Upgrading existing approvals

Stop the node, its GUI and any other database writers, then back up the node database before upgrading. On the first use of the training-plan manager, existing automatic `default` records become ordinary `requested` records. Approved or pending defaults require a new explicit review; existing rejections remain rejected. Stored source, identifiers, notes and dates are retained, and the migration is idempotent.

Hashes made with older algorithms are converted to SHA256 only after verifying the stored source against its recorded legacy digest. Explicit approvals and rejections for other plans are preserved. The migration validates all records before updating the table in one write; this is not a crash-safe transaction. If source is missing or does not match, an algorithm is unknown, or two records become the same code hash, the manager stops without applying the migration. Reconcile those records using a database backup and the previous release before retrying; do not infer approval for unverifiable source.

The retired `hashing_algorithm` and `allow_default_training_plans` settings are removed from the loaded configuration. The old default-plan environment variable has no effect. The independent `training_plan_approval` switch keeps its configured value, including its default `False`.

### Using Fed-BioMed CLI

Enable approval mode when creating or starting a node:

```shell
FBM_SECURITY_TRAINING_PLAN_APPROVAL=True fedbiomed node --path ./my-node start
```

With approval mode enabled, every tutorial or custom plan must be approved on the node before it can run. Use the registration and approval commands below; enabling approval mode alone does not approve any plan.

## Training Plan Registration and Approval

The training plan registration and approval process is done by the Fed-BioMed CLI or GUI tool.

Fed-BioMed training plans have one of the following types on a node:

* **requested training plans** are sent by the researcher to nodes from inside the application ("in band"). This enables the researcher to easily submit a training plan to nodes for approval. This mode is the most commonly used for having an `Experiment`'s training plan approved on nodes.
* **registered training plans** are manually added on this node from a file ("out of band"). This enables the node to use a training plan from any source.

Fed-BioMed training plans have one of the following status:

* **Approved** training plans are authorized to train and test on this node.
* **Pending** training plans are waiting for review and approval/rejection decision on this node.
* **Rejected** training plans are explicitly not authorized to run on this node.

Training Plans are saved in the database with following details:

```json
{
    'name' : '<training-plan-name>',
    'description' : '<description>',
    'training_plan_type' : 'registered',
    'training_plan_path' : '<path/to/training-plan/fıle>',
    'training_plan_id' : '<Unique id for the training plan>',
    'researcher_id' : '<The ID of the researcher that sent this training plan or None>',
    'algorithm' : '<algorithm used for the hash of the training plan file>',
    'hash' : '<hash of the training plan file>',
    'date_registered' : '<Registration date>',
    'date_created' : '<The date file has been created>',
    'date_modified' : '<The date file has been modified>',
    'date_last_action' : '<The date file has been modified or hash recomputed>'
}
```

Note: training plan files are stored in the file system as `txt` files. Input training plan files used as
registered training plans must have this format.

### Using requested training plans

Requested training plans are training plans which are sent by a researcher to nodes, inside the Fed-BioMed software.

After defining an `Experiment()` named `exp` in a researcher's notebook, the following command is typed in the notebook to send the requested training plan's training plan to the nodes of `exp`:

```python
#exp = Experiment(...
#   training_plan_class=MyTrainingPlan,
#   ...)

exp.training_plan_approve(MyTrainingPlan, description="A human readable description of the training plan")
```

When receiving the training plan, `exp`'s nodes register the training plan in their persistent database with a type `registered`, a `Pending` status, and the `description` sent by the researcher. If a training plan with same hash already exists in the database, nothing is added and an error is returned to the researcher.

A human reviewer then checks and decides whether the training plan should be authorized on the node, via the GUI or the CLI.

Use the following command to view a training plan on the CLI, optionally indicating your preferred editor with `EDITOR`. All registered training plans are listed with their name and status, select the one you want to view:

```shell
# EDITOR=emacs fedbiomed node --path ./my-node training-plan view

$ fedbiomed node --path ./my-node training-plan view
```

After reviewing the training plan, use one of the following commands to either approve or reject the training plan:

```shell
$ fedbiomed node --path ./my-node training-plan approve

$ fedbiomed node --path ./my-node training-plan reject
```

The command returns the list of training plans that can be approved/rejected, choose the reviewed training plan from the list:

```shell
Select the training plan to approve:
1) training_plan_5fa329d7-af27-4461-b7e7-87e5b8b5e7b6	 Model ID training_plan_5fa329d7-af27-4461-b7e7-87e5b8b5e7b6  training-plan status Pending	date_last_action None
2) training_plan_281464db-ab53-494a-bd58-951957eee762    Model ID training_plan_281464db-ab53-494a-bd58-951957eee762  training-plan status Pending	date_last_action None
Select: 1
```

Training Plan status on the node can later be changed again using the same `training-plan approve` and `training-plan reject` commands.

When a node doesn't want to keep track of a registered training plan anymore, it can delete it from the database.
The command does not delete the file containing the modtraining planel, only the database entry for the training plan.

```shell
$ fedbiomed node --path ./my-node training-plan delete
```

The output of this command lists deletable training plans with their names and id. It asks you to select the training plan file you would like to remove. For example, in the following example, typing 1  removes the MyTrainingPlan from registered/approved
list of training plans.

```shell
Select the training plan to delete:
1) MyTrainingPlan   Training Plan ID training_plan_98a1e68d-7938-4889-bc46-357e4ce8b6b5
2) MyTrainingPlan2  Training Plan ID training_plan_18314625-2134-3334-vb35-123f3vbe7fu7
Select:
```


### Using Registered training plans

Registered training plans are training plans manually added to the node via the GUI or the CLI, from a file containing its training plan.

The following command launches Fed-BioMed CLI for selecting a training plan file and entering a name and description for the training plan. The training plan name, its path and its hash should be unique. It means that you can not add the same training plan file multiple times.

```shell
$ fedbiomed node --path ./my-node training-plan register
```
After selecting the training plan file, the training plan manager computes a hash for the training plan file and saves it into the persistent database.

The training plan type is `registered`, and status is `Approved` for the training plans that are saved through Fed-BioMed CLI.

Each time that the node is started, training plan manager checks whether the training plan file still exists on the file system. If it is deleted,
training plan manager also deletes it from the database. Therefore, please make sure that the training plan file always exists in the path where
it is stored.

As for requested training plans, registered training plans can later be viewed (`training-plan viw`), changed status (`training-plan approve` or `training-plan reject`) or removed (`training-plan delete`).

It is also possible to update registered training plans with a different file or the same training plan file whose content has changed. This is useful when working on a training plan, and you want it to be updated without having to remove it and restore it in database. The following command launches the CLI to select the training plan that will be updated

```shell
$ fedbiomed node --path ./my-node training-plan update
```

The command lists registered training plans with their names and ids and asks you to select a training plan you would like to update. Afterward, it asks to select a training plan file from file system. You can either select a different or the same training plan file. It computes a hash for the specified training plan file and updates the database.

!!! note
    You can update hashes only by providing a training plan file. This API does not allow you to update saved
    hashes directly.
