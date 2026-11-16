# SAGE & ML Commons & Fed-BioMed: COVID-CT classification

Federated binary classification (COVID vs NonCOVID) of chest CT slices over two
hospitals with a realistic non-IID split. The use case is small enough to run
on CPU. It works with the capability guardian described in `README.md`.

Files in this folder:

| File | Role |
|---|---|
| `download_and_split_covidct.py` | downloads the dataset, creates the `hospital-a` / `hospital-b` node components and their image folders |
| `covidct-classification-training.ipynb` | training notebook (+ holdout evaluation, optional capability) |
| `covidct_one_node_example.py` | the same experiment as a script, handy for smoke tests and guardian scenarios |

## Dataset

[COVID-CT](https://arxiv.org/abs/2003.13865) (UCSD-AI4H): 349 CT slices with
COVID-19 findings, from 216 patients, extracted from 57 publications. It also has
397 NonCOVID slices from 2 sources: 202 crawled from the literature and 195 from
the MedPix database. The original GitHub repository (`UCSD-AI4H/COVID-CT`) is no
longer online, so the script downloads from the mirror
`zziyifan/COVID-CT` (about 90 MB, MD5-checked). Use `--base-url` to point to another
copy with the same layout. All copyrights of the images belong to the authors
and publishers of the source papers. The dataset is for research use only.

Zhao J. et al., *COVID-CT-Dataset: A CT Scan Dataset about COVID-19*, arXiv:2003.13865, 2020.

## 1. Download and split

The script needs `requests`, `numpy`, `Pillow`, `scikit-learn` and `tqdm`, which
are already installed with Fed-BioMed. Run it from the directory where the two
node folders should be created:

```bash
mkdir -p ./covid-ct-data
python download_and_split_covidct.py -f ./covid-ct-data
```

| Flag | Default | Meaning |
|---|---|---|
| `-f/--root_folder` | required | **existing** folder the raw archives are downloaded to (`<root>/covid-ct/`) |
| `-F/--force` | off | re-create the node components even if they already exist |
| `--covid-share` | `0.75` | fraction of COVID slices assigned to hospital-a |
| `--noncovid-mix` | `0.15` | fraction of each NonCOVID source's patients sent to the *other* hospital |
| `--holdout` | `0.15` | holdout fraction inside each hospital (by group) |
| `--image-size` | `224` | output slices are grayscale, padded to a square and resized |
| `--base-url` | mirror | where to download the files from |
| `--seed` | `21` | seed for the assignment and the holdout split |

The `data` folders are always rebuilt. Existing node components are kept
unless you pass `-F`.

### How the non-IID split works

The split follows where the images come from, as it would between two real
institutions. A random draw would not give that.

- **COVID** slices are grouped by **source publication**. Each publication
  reports patients of one or a few hospitals. Whole publications go to a node
  until hospital-a holds `--covid-share` of the COVID slices.
- **NonCOVID** slices crawled from the literature go to hospital-a, and MedPix
  slices go to hospital-b. A `--noncovid-mix` fraction of patients from each
  source is swapped to the other node.
- Groups (publications, NonCOVID patients) are never split: the same patient
  never appears on both nodes, or in both train and holdout.

The result is **label skew**, **quantity skew** and **feature shift** (different
image provenance). With the defaults:

```
node        split     COVID  NonCOVID  sources
hospital-a  train       230       157  crawled, medpix, publication
hospital-a  holdout      33        46  crawled, publication
hospital-b  train        70       161  crawled, medpix, publication
hospital-b  holdout      16        33  crawled, medpix, publication
```

That makes hospital-a 56% COVID with 466 slices, and hospital-b 31% COVID with 280 slices.

Each node gets this layout:

```
hospital-a/
├── etc/ var/ ...                 # node component
├── dataset_train.json            # description files for `dataset add --file`
├── dataset_holdout.json
└── data/
    ├── train/
    │   ├── CT_COVID/*.png
    │   └── CT_NonCOVID/*.png
    └── holdout/
        ├── CT_COVID/*.png
        └── CT_NonCOVID/*.png
```

## 2. Add the data to each node as an image-folder dataset

Fed-BioMed's **images** data type is backed by `ImageFolderDataset`. It reads a
root folder holding one sub-folder per class:

- the root folder contains **only** class folders. An image directly in the
  root, or nested sub-folders inside a class folder, breaks loading;
- class labels come from the folder names, **sorted alphabetically**:
  `CT_COVID` → 0, `CT_NonCOVID` → 1;
- every node must have the **same class folders**. Otherwise the label indices
  differ between nodes;
- images can be any format PIL reads. They are loaded as RGB and given to the
  training plan's transform as `3 x H x W` float tensors in `[0, 1]`.

### Non-interactive (recommended)

```bash
fedbiomed node -p hospital-a dataset add --file hospital-a/dataset_train.json
fedbiomed node -p hospital-b dataset add --file hospital-b/dataset_train.json
```

The description file is plain JSON:

```json
{
  "name": "covid-ct-train",
  "path": "/abs/path/to/hospital-a/data/train",
  "data_type": "images",
  "tags": "covid-ct",
  "description": "COVID-CT train split of hospital-a"
}
```

`tags` is a **comma-separated string** (e.g. `"covid-ct,ct"`), not a JSON list.
A relative `path` is resolved against the node component folder.


## 2. Start the nodes

```bash
fedbiomed node -p hospital-a start
fedbiomed node -p hospital-b start
```

Leave each one running in its own terminal. New components accept any training
plan (`training_plan_approval = False`). If approval is enabled, approve the plan
with `fedbiomed node -p <node> training-plan approve`.

## 3. Run the experiment

Create a researcher component and start Jupyter from this folder:

```bash
fedbiomed component create -c researcher
fedbiomed researcher start
```

Open `covidct-classification-training.ipynb` and run it. The training plan:

- **model:** ResNet-18 from `torchvision`, randomly initialised (`weights=None`, so
  nodes download nothing), with a 2-class head;
- **data:** `ImageFolderDataset` with `Resize(224)`, `RandomHorizontalFlip` and
  `Normalize`, all tensor-compatible, because transforms receive tensors;
- **loss:** cross-entropy. Validation on 15% of each node's train folder uses
  `MetricTypes.ACCURACY` on the global model at every round.

Alternatively, run the script:

```bash
python covidct_one_node_example.py --rounds 5                # all covid-ct nodes
python covidct_one_node_example.py --rounds 2 --batch-maxnum 5   # quick smoke test
```

It works with one node (only `hospital-a` started) as well as with both.
Every node connected to the researcher with a `covid-ct` dataset takes part, so
stop the nodes you do not want in the run.

The script's training plan also has a `testing_step`. It reports the
validation loss, accuracy, sensitivity, specificity and F1, with COVID as the
positive class. After training, the script:

- prints one table per node with these metrics for the global model after each
  aggregation;
- saves a figure to `--plot-file` (default `covidct_training_history.png`).
  It shows the training loss of each node (faint: reported batches, bold:
  round mean), the validation loss of the global model, and one performance
  panel per node. Add `--show` to also open it in a window.

The values come from the researcher's monitor
(`exp.monitor().metric_store()`). Training loss is reported every
`log_interval` (2) batches. Validation runs once per round on the node's whole
validation split. The last batch of a round can be small, so the faint
per-batch line shows spikes.

### Holdout evaluation

The last notebook section validates the trained model on the `covid-ct-holdout`
datasets with `test_ratio=1.0`. A known limitation: the node reports the
holdout accuracy, then the round ends with
`FB300 ... 'NoneType' object has no attribute 'dataset'`, because the train split
is empty (`FIXME` in `fedbiomed/node/round.py`). The accuracy printed before
the error is valid.

## 5. With the capability guardian

The plan is self-contained (`init_dependencies` lists every import), so its
digest is stable. Follow *Enabling the capability guardian* in `README.md`
(start the mock guardian, set `guardian_service` on the nodes), then:

```bash
python covidct_one_node_example.py --capability valid     # rounds run
python covidct_one_node_example.py --capability invalid   # guarded nodes refuse
python covidct_one_node_example.py --capability none      # no check
```

With a capability, the script prints its content before training: the
authorized digest, the method and the decoded package (long random fields
shortened). The same output comes from `python mint_capability.py <digest>`, or
from `format_capability(capability)` in Python. A real capability's operation
is encrypted, so only its envelope is shown.

```
Capability package (1120 bytes, base64)
  method:              do_inference
  authorized digest:   sha256:518b0de4...
  request identifier:  /6BhhFgQ4uU=
  full package:
    {
      "minted_identity": "cHmFYEjgmVJXm4DEA6CyfP58... (44 chars)",
      "operation": {
        "encrypted_session_key": "Eq6WmISRA6qlLfxs0gWsJPza... (344 chars)",
        "session_key_iv": "gT9kTm7lZdsDujiuuVSAAQ==",
        "encrypted_message": {
          "nonce": "...",
          "request_identifier": "/6BhhFgQ4uU=",
          "method_name": "do_inference",
          "parameters": {"script_digest": "sha256:518b0de4..."}
        }
      }
    }
```

If the node cannot reach the guardian with a `ProxyError ... host='https'`,
macOS has a proxy configured that Python applies to `localhost` too. Start the
node with `NO_PROXY=localhost,127.0.0.1` (and `no_proxy`) set.

In the notebook, use the last cell (`exp.set_capabilities(mint_capability(digest))`)
before `exp.run()`.

## Tested

A single-node run on `hospital-a` (CPU, default split) gave:

- `--rounds 2 --batch-maxnum 5`: both rounds completed;
- `--rounds 5`: global validation accuracy went 0.52 → 0.64 → 0.62 → 0.67 → 0.53 → 0.55.
  The model is trained from scratch on about 330 slices, so expect noisy curves.
  More rounds, both nodes, or a larger learning rate help.
