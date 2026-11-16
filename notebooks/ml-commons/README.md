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

## Workflow

```
Experiment(capabilities="<base64 blob>")
        │
TrainRequest.capabilities  ──────>  Node
                                      │
                                      │ 1. computes "sha256:" + sha256(training_plan_source)
                                      │ 2. POST {guardian_service}/process_capability
                                      │      {"capability_b64": "<blob>",
                                      │       "request_context": {"calculated_script_digest": "sha256:..."}}
                                      │
                              Guardian service
                                      │
                    200 = allowed, 422 = refused, other/unreachable = error
                                      │
                          ┌───────────┴───────────┐
                  allowed -> train/test    refused or error -> round refused,
                                           training plan is never imported
```

1. A third party (policy engine) issues a **capability** for a training plan. It
   is an opaque base64 string: real capabilities are encrypted, and neither the
   researcher nor the node ever parses it. It authorizes one `script_digest`,
   the `sha256:<hex>` digest of the training plan source.
2. The researcher passes that string to `Experiment(capabilities=...)`. It is
   attached, unmodified, to every `TrainRequest` sent to the nodes.
3. Each node that declares a `guardian_service` in its configuration:
   - computes the `sha256:<hex>` digest of the training plan source it
     actually received,
   - calls `POST {guardian_service}/process_capability` with the capability
     and that digest,
   - proceeds with training and validation only on a success answer. A refusal
     (`422`), any other status, or an unreachable guardian all refuse the round
     (fail closed).
4. A node with **no** `guardian_service` configured skips this check entirely,
   so existing deployments are unaffected. A request with no capability is
   skipped as well.

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

### 1. Start the mock guardian service

`mock_inference_guardian/` holds a PoC-only copy of the policy team's mock
inference guardian (see its `README.md` for where it comes from). It answers
`POST /process_capability` like the real guardian: allowed when the
capability's `script_digest` equals the digest the node reports, `422`
otherwise. It does no decryption or signature verification, so do not deploy it.

```bash
python mock_inference_guardian/server.py --port 7900
```

or, in Docker:

```bash
mock_inference_guardian/build.sh && mock_inference_guardian/run.sh --port 7900
```

To check it on its own:

```bash
cd mock_inference_guardian
python mint_capability.py sha256:abc               # -> ./capability.b64
python redeem.py -f ./capability.b64 -c sha256:abc   # HTTP 200; any other -c -> 422
```

### 2. Point the node at it

Either in the node's `etc/config.ini` under `[security]` (e.g. `guys/etc/config.ini`):

```ini
[security]
guardian_service = http://localhost:7900
```

or via environment variable when creating/starting the node:

```shell
export FBM_SECURITY_GUARDIAN_SERVICE=http://localhost:7900
fedbiomed node -p guys start
```

A node running in a container reaches a guardian on the host at
`http://host.docker.internal:7900`. Leaving `guardian_service` empty (the
default) disables capability validation on that node.

### 3. Pass a capability to the Experiment

The digest must be computed over the exact source the nodes receive. Take it
from the Experiment's own training plan, not from a fresh
`UNetValidationPlan()`: the dependency imports are only added to the source
once the Experiment has set the training plan up. Then mint the capability
for that digest and attach it:

```python
import hashlib
import sys

from fedbiomed.researcher.aggregators.fedavg import FedAverage
from fedbiomed.researcher.federated_workflows import Experiment

sys.path.append("./mock_inference_guardian")
from mint_capability import mint_capability

exp = Experiment(
    tags=["brain-segmentation"],
    model_args=model_args,
    training_plan_class=UNetValidationPlan,
    training_args=training_args,
    round_limit=num_rounds,
    aggregator=FedAverage(),
)

source = exp.training_plan().source()
digest = "sha256:" + hashlib.sha256(source.encode("utf-8")).hexdigest()

exp.set_capabilities(mint_capability(digest))
exp.run()
```

With a real policy engine, the capability comes from the engine instead of
`mint_capability`, and is passed the same way, as a string. It can also be
given directly to `Experiment(capabilities=...)`. The capability survives
`exp.breakpoint()` / `Experiment.load_breakpoint()`, and is re-sent on the
final validation-only round when `run(test_after=True)` is used. A node checks
that round too, not just training rounds.

### Trying it end to end

1. Start the mock guardian on port 7900 (step 1).
2. Start the node with `guardian_service = http://localhost:7900` (step 2).
3. Run the training notebook with a capability minted for the training plan's
   digest (step 3). Training runs, and the guardian logs
   `releasing data for authorized script sha256:...`.
4. Mint the capability for another digest instead, e.g.
   `mint_capability("sha256:0000")`, and rerun. The node refuses every round
   (`success=False`, "operation failed": no training or testing ever runs), and
   the guardian logs `script digest mismatch`.
5. Stop the guardian and rerun. The node refuses the round with
   "Could not validate capability with the guardian service" (fail closed).
6. Unset `guardian_service` on the node and rerun with the mismatched
   capability. The node skips the check and trains normally: the check is
   permissive only when no guardian is configured.
