# SAGE & ML Commons & Fed-BioMed: brain segmentation + capability guardian

This folder holds two pieces:

- a brain segmentation example (`download_and_split_ixi.py`,
  `brain-segmentation-training.ipynb`,
  `brain-segmentation-testing-with-pretrained-model.ipynb`) that trains and
  evaluates a UNet on the IXI dataset;
- the wiring needed to run that experiment under **capability-based policy
  gating**: the researcher attaches a signed capability to the experiment, and
  each node checks it against a local **guardian service** before it will
  train or validate anything.

The two are independent — the notebooks run fine with no guardian configured
at all. "Running the brain-segmentation experiment" below is the plain setup;
"Enabling the capability guardian" adds the policy gate on top.

> **Current status:** the actual HTTP call in `GuardianClient.verify()`
> (`fedbiomed/node/guardian.py`) is temporarily commented out and always
> returns `valid=True` — see the "Guardian service verification is currently
> disabled" note in that file. The wiring described below (config, checksum
> computation, `Experiment(capabilities=...)`) is all active; only the actual
> network round-trip and its answer are short-circuited until it's turned
> back on.

## Workflow

```
Experiment(capabilities={...})
        │
TrainRequest.capabilities  ──────>  Node
                                      │
                                      │ 1. computes sha256(training_plan_source)
                                      │ 2. POST {guardian_service}/verify
                                      │      {capabilities, training_plan_checksum, ...}
                                      |
                              Guardian service
                                      │
                              {"valid": true/false, "reason": "..."}
                                      │
                          ┌───────────┴───────────┐
                  valid -> train/test      not valid -> round refused,
                                           training plan is never imported
```

1. A third party (policy engine) signs a training plan and issues a **capability**, for now,
   in this codebase, just `{"training_plan_checksum": "<sha256 hex>", ...}`.
   There is no issuance flow yet; the researcher hard-codes it.
2. The researcher passes that capability to `Experiment(capabilities=...)`. It
   is attached, unmodified, to every `TrainRequest` sent to the nodes.
3. Each node that declares a `guardian_service` in its configuration:
   - computes `sha256` of the training plan source it actually received,
   - calls `POST {guardian_service}/verify` with the capability and that
     checksum,
   - proceeds with training and validation only if the answer is
     `{"valid": true}`. A rejection, a malformed answer, or an unreachable
     guardian all refuse the round (fail closed).
4. A node with **no** `guardian_service` configured skips this check entirely
   — existing deployments are unaffected. A capability with no configured
   guardian, or a request with no capability at all, is likewise skipped.

See `fedbiomed/node/guardian.py` for the client and
`fedbiomed/node/round.py` (`Round._verify_capabilities`) for where the gate is
applied, right before the training plan is imported.


## Running the brain-segmentation experiment

These steps stand up three federated nodes (`Guys`, `HH`, `IOP`) serving the
IXI brain-MRI data, then run the training notebook against them. No guardian is
involved here — see "Enabling the capability guardian" for that.

Both the nodes and the researcher need the training-plan dependencies:

```bash
pip install unet monai tqdm
```

### 1. Download and split the IXI dataset

`download_and_split_ixi.py` downloads the IXI sample dataset (from Mendeley
Data), then — **in the current working directory** — creates one node
component per hospital centre (`./guys`, `./hh`, `./iop`) and fills each with a
90/10 `data/train` + `data/holdout` split of that centre's subjects. Every
split folder also gets a `participants.csv` holding the subject demographics.

```bash
mkdir -p ./ixi-data
python download_and_split_ixi.py -c -f ./ixi-data
```

- `-f/--root_folder` — an **existing** directory the raw archive is downloaded
  into and extracted.
- `-c/--centralized-data-folder` — required flag; use the `-f` folder directly
  as the download target (without it the script expects a `notebooks/data`
  subfolder instead).
- `-F/--force` — re-create the `guys`/`hh`/`iop` node components even if they
  already exist (otherwise existing components are left untouched).

Run it from the directory where you want the three node folders to live. When
it finishes it prints the exact `dataset add` and `node start` commands for the
next two steps.

### 2. Register the IXI dataset on each node

For each centre, run the interactive dataset wizard:

```bash
fedbiomed node -p guys dataset add
fedbiomed node -p hh   dataset add
fedbiomed node -p iop  dataset add
```

Answer the prompts the same way for every node:

| Prompt | Answer |
|---|---|
| Data type | `5` (`medical-folder`) |
| Name of the database | any name ≥ 3 chars, e.g. `ixi` |
| Tags | `brain-segmentation` — must match the `tags` list in the notebook |
| Description | any text ≥ 3 chars |
| Root folder of the Medical Folder dataset | `guys/data/train` (resp. `hh/data/train`, `iop/data/train`) |
| Would you like to select a demographics csv file? | `y` |
| Demographics file | `guys/data/train/participants.csv` |
| Index of the subject-id column | `14` (the column holding the image-folder name) |

To evaluate on held-out subjects instead, point at the `data/holdout` folders
and change the notebook's `tags` to match (e.g. `ixi-holdout`).

### 3. Start the nodes

```bash
fedbiomed node -p guys start
fedbiomed node -p hh   start
fedbiomed node -p iop  start
```

Leave each one running in its own terminal.

### 4. Run the training notebook

Create a researcher component and launch Jupyter from this folder (so the
`unet` import and the notebook resolve):

```bash
fedbiomed component create -c researcher
fedbiomed researcher start
```

Open `brain-segmentation-training.ipynb` and run it; its `tags` list must match
the tag registered in step 2. For the pretrained-model path use
`brain-segmentation-testing-with-pretrained-model.ipynb` together with
`brain-segmentation.pt`.

## Enabling the capability guardian

### 1. Start a node with the guardian enabled

First run a guardian service. A development stub that only compares the two
checksums (no real signature verification) ships at
`scripts/mock_guardian_server.py`.

> Note: no guardian service is running by default, and the HTTP call in
> `GuardianClient.verify()` is currently short-circuited (see the status note
> at the top), so this section is wiring-only for now.

Then point the node at it, either in the node's `etc/config.ini` under
`[security]` (e.g. `guys/etc/config.ini`):

```ini
[security]
guardian_service = http://localhost:8000
```

or via environment variable when creating/starting the node:

```shell
export FBM_SECURITY_GUARDIAN_SERVICE=http://localhost:8000
fedbiomed node start
```

Leaving `guardian_service` empty (the default) disables capability validation
on that node.

### 2. Pass a capability to the Experiment

The researcher computes the same checksum the node will compute — a plain
SHA-256 of the training plan's source — and wraps it in a capability dict:

```python
import hashlib
from fedbiomed.researcher.federated_workflows import Experiment
from fedbiomed.researcher.aggregators.fedavg import FedAverage

training_plan_source = UNetValidationPlan().source()
checksum = hashlib.sha256(training_plan_source.encode()).hexdigest()


# Note: There is no check for the capabilities object schema
capabilities = {
    "training_plan_checksum": checksum,
    "signature": "<issued by the third party>",
    "issuer": "<authority id>",
}

exp = Experiment(
    tags=["brain-segmentation"],
    model_args=model_args,
    training_plan_class=UNetValidationPlan,
    training_args=training_args,
    round_limit=num_rounds,
    aggregator=FedAverage(),
    capabilities=capabilities,
)

exp.run()
```

That's the only change needed on top of the existing
`brain-segmentation-training.ipynb` cell that builds `exp`. The capability
also survives `exp.breakpoint()` / `Experiment.load_breakpoint()`, and is
re-sent on the final validation-only round when `run(test_after=True)` is
used — a node checks that round too, not just training rounds.

### Trying it end to end

While the HTTP call is disabled (see the status note above), every node with
a `guardian_service` configured accepts every request unconditionally. There
is no wrong-checksum case to observe yet. Once `GuardianClient.verify()` is
back to actually calling the service, this is the sequence to exercise the
full gate:

1. (optional not tested yet) `python scripts/mock_guardian_server.py --port 8000`
2. Start the node with `guardian_service = http://localhost:8000`.
3. Run the training notebook with `capabilities` set to the checksum of
   `UNetValidationPlan` (see "2. Pass a capability to the Experiment")
4. Change one character in `capabilities["training_plan_checksum"]` and rerun. The node 
   refuses every round (`success=False`, no training or testing ever runs), and the guardian prints `valid=False`.
5. Unset `guardian_service` on the node and rerun with the same (now
   mismatched) capability. The node skips the check and trains normally,
   confirming the fallback is permissive only when no guardian is configured.

`scripts/mock_guardian_server.py` is a development stub, not a policy
authority: it does no signature verification, only checksum comparison. Do
not deploy it.
