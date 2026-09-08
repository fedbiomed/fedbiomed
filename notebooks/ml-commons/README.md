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
at all (see "Workflow" below). This README only covers the guardian/capability
part; for the segmentation experiment itself see the notebooks.

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

## 1. Start a node with the guardian enabled

First run a guardian service. A development stub that only compares the two
checksums (no real signature verification) ships at
`scripts/mock_guardian_server.py`:


Note: Currently there is no guardian service is active. 

Then point the node at it, either in `etc/config.ini` under `[security]`:

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

## 2. Pass a capability to the Experiment

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

## Trying it end to end

While the HTTP call is disabled (see the status note above), every node with
a `guardian_service` configured accepts every request unconditionally — there
is no wrong-checksum case to observe yet. Once `GuardianClient.verify()` is
back to actually calling the service, this is the sequence to exercise the
full gate:

1. (optional not tested yet) `python scripts/mock_guardian_server.py --port 8000`
2. Start the node with `guardian_service = http://localhost:8000`.
3. Run the training notebook with `capabilities` set to the checksum of
   `UNetValidationPlan` (see 2)
4. Change one character in `capabilities["training_plan_checksum"]` and rerun. The node 
   refuses every round (`success=False`, no training or testing ever runs), and the guardian prints `valid=False`.
5. Unset `guardian_service` on the node and rerun with the same (now
   mismatched) capability. The node skips the check and trains normally,
   confirming the fallback is permissive only when no guardian is configured.

`scripts/mock_guardian_server.py` is a development stub, not a policy
authority: it does no signature verification, only checksum comparison. Do
not deploy it.
