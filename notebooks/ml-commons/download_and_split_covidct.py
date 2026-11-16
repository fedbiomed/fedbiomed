#!/usr/bin/env python
"""COVID-CT downloader and non-IID splitter for two Fed-BioMed nodes.

Downloads the UCSD-AI4H COVID-CT dataset (349 COVID / 397 NonCOVID CT slices),
then, in the current working directory, creates two node components
(`./hospital-a`, `./hospital-b`) and fills each with an image-folder dataset:

    hospital-x/data/{train,holdout}/{CT_COVID,CT_NonCOVID}/*.png

The split follows the provenance of the images rather than a random draw:

- COVID slices are grouped by the publication they were extracted from (each
  publication reports patients of one or a few hospitals); whole publications
  are assigned to a node, `--covid-share` of the COVID slices going to
  hospital-a.
- NonCOVID slices come from two sources: images crawled from the literature
  (go to hospital-a) and the MedPix database (go to hospital-b);
  `--noncovid-mix` of each source's patients are swapped to the other node.

Both nodes thus differ in label ratio, size and image provenance. Slices of a
same publication/patient never end up on both nodes, nor in both the train and
the holdout split of a node.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
from zipfile import ZipFile

import numpy as np
import requests
from PIL import Image, ImageOps
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

from fedbiomed.node.config import node_component

# The original UCSD-AI4H/COVID-CT repository is no longer online, the files are
# taken from a mirror with the same layout.
DEFAULT_BASE_URL = "https://raw.githubusercontent.com/zziyifan/COVID-CT/master"

FILES = {
    "CT_COVID.zip": (
        "Images-processed/CT_COVID.zip",
        "302d5a9df44321a6d4dce24db6fada89",
    ),
    "CT_NonCOVID.zip": (
        "Images-processed/CT_NonCOVID.zip",
        "7d372187a33ae18d1bb5ec63c7cc65cf",
    ),
    "NonCOVID-CT-MetaInfo.csv": ("NonCOVID-CT-MetaInfo.csv", None),
}

CLASSES = ["CT_COVID", "CT_NonCOVID"]
NODES = ["hospital-a", "hospital-b"]


def parse_args():
    parser = argparse.ArgumentParser(description="COVID-CT downloader and splitter")
    parser.add_argument(
        "-f",
        "--root_folder",
        required=True,
        type=str,
        help="Existing folder the raw dataset is downloaded into",
    )
    parser.add_argument(
        "-F",
        "--force",
        action=argparse.BooleanOptionalAction,
        required=False,
        default=False,
        help="Re-create node components and data folders if they already exist",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, type=str)
    parser.add_argument(
        "--covid-share",
        default=0.75,
        type=float,
        help="Fraction of COVID slices assigned to hospital-a",
    )
    parser.add_argument(
        "--noncovid-mix",
        default=0.15,
        type=float,
        help="Fraction of each NonCOVID source's patients sent to the other node",
    )
    parser.add_argument("--holdout", default=0.15, type=float)
    parser.add_argument("--image-size", default=224, type=int)
    parser.add_argument("--seed", default=21, type=int)
    return parser.parse_args()


def has_correct_checksum_md5(filename, hash):
    with open(filename, "rb") as f:
        file_hash = hashlib.md5()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return str(file_hash.hexdigest()) == hash


def download_file(url, filename):
    """Download `url` to `filename`, raising requests.HTTPError on error status."""
    print(f"Downloading {url}")
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    content_length = r.headers.get("Content-Length")
    total = int(content_length) if content_length is not None else None
    with open(filename, "wb") as f:
        pbar = tqdm(unit="B", total=total, unit_scale=True)
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                pbar.update(len(chunk))
                f.write(chunk)
    return filename


def download_and_extract(data_folder, base_url):
    """Downloads and extracts the dataset files into `data_folder/covid-ct`.

    Returns:
        raw_folder: folder holding `CT_COVID/`, `CT_NonCOVID/` and the metadata
    """
    raw_folder = os.path.join(data_folder, "covid-ct")
    os.makedirs(raw_folder, exist_ok=True)

    for filename, (remote_path, md5) in FILES.items():
        local = os.path.join(raw_folder, filename)
        if not os.path.exists(local):
            download_file(f"{base_url}/{remote_path}", local)
        if md5 is not None and not has_correct_checksum_md5(local, md5):
            raise ValueError(
                f"MD5 checksum mismatch for '{local}'. Delete it and re-run the script."
            )

    for class_name in CLASSES:
        if not os.path.isdir(os.path.join(raw_folder, class_name)):
            print(f"Extracting {class_name}.zip")
            with ZipFile(os.path.join(raw_folder, f"{class_name}.zip")) as zip_obj:
                members = [
                    m
                    for m in zip_obj.namelist()
                    if m.startswith(f"{class_name}/") and not m.endswith("/")
                ]
                zip_obj.extractall(raw_folder, members=members)

    return raw_folder


def covid_publication(filename):
    """Returns the publication a COVID slice was extracted from.

    File names look like `2020.03.01.20029769-p21-73_1%1.png` (medRxiv id, page,
    figure, slice) or `PIIS0140673620301549_0%1.png` (journal article id).
    """
    stem = os.path.splitext(filename)[0]
    return re.split(r"-p\d+|%|_\d", stem)[0]


def list_samples(raw_folder):
    """Lists the slices with their class, source and group.

    Returns:
        samples: list of dicts with keys `file`, `label`, `source`, `group`
    """
    samples = []
    covid_dir = os.path.join(raw_folder, "CT_COVID")
    for name in sorted(os.listdir(covid_dir)):
        if name.startswith("."):
            continue
        publication = covid_publication(name)
        samples.append(
            {
                "file": os.path.join(covid_dir, name),
                "label": "CT_COVID",
                "source": "publication",
                "group": f"covid/{publication}",
            }
        )

    noncovid_dir = os.path.join(raw_folder, "CT_NonCOVID")
    with open(os.path.join(raw_folder, "NonCOVID-CT-MetaInfo.csv")) as f:
        for row in csv.DictReader(f):
            name, patient = row["image name"], row["patient id"]
            path = os.path.join(noncovid_dir, name)
            if not os.path.isfile(path):
                continue
            samples.append(
                {
                    "file": path,
                    "label": "CT_NonCOVID",
                    "source": "medpix" if patient.startswith("medpix") else "crawled",
                    "group": f"noncovid/{patient}",
                }
            )
    return samples


def assign_nodes(samples, covid_share, noncovid_mix, rng):
    """Assigns every group of slices to a node.

    Returns:
        node_of_group: dict mapping a group name to a node name
    """
    node_of_group = {}

    # COVID: whole publications, until hospital-a holds `covid_share` of slices
    sizes = {}
    for s in samples:
        if s["label"] == "CT_COVID":
            sizes[s["group"]] = sizes.get(s["group"], 0) + 1
    groups = list(sizes)
    rng.shuffle(groups)
    target_a = covid_share * sum(sizes.values())
    count_a = 0
    for group in groups:
        if count_a + sizes[group] / 2 <= target_a:
            node_of_group[group] = NODES[0]
            count_a += sizes[group]
        else:
            node_of_group[group] = NODES[1]

    # NonCOVID: crawled -> hospital-a, medpix -> hospital-b, with some swapped
    home = {"crawled": NODES[0], "medpix": NODES[1]}
    for source, node in home.items():
        other = NODES[1] if node == NODES[0] else NODES[0]
        groups = sorted(
            {
                s["group"]
                for s in samples
                if s["label"] == "CT_NonCOVID" and s["source"] == source
            }
        )
        rng.shuffle(groups)
        n_swap = int(round(noncovid_mix * len(groups)))
        for i, group in enumerate(groups):
            node_of_group[group] = other if i < n_swap else node

    return node_of_group


def split_holdout(node_samples, holdout, seed):
    """Splits a node's slices into train/holdout by group, class by class.

    Returns:
        train: slices of the train split
        test: slices of the holdout split
    """
    train, test = [], []
    for class_name in CLASSES:
        items = [s for s in node_samples if s["label"] == class_name]
        if len({s["group"] for s in items}) < 2:
            train.extend(items)
            continue
        splitter = GroupShuffleSplit(n_splits=1, test_size=holdout, random_state=seed)
        idx_train, idx_test = next(
            splitter.split(items, groups=[s["group"] for s in items])
        )
        train.extend(items[i] for i in idx_train)
        test.extend(items[i] for i in idx_test)
    return train, test


def save_image(src, dst, image_size):
    """Saves `src` as a square grayscale PNG of side `image_size`."""
    image = Image.open(src).convert("L")
    image = ImageOps.pad(image, (image_size, image_size), color=0)
    image.save(dst)


def write_split(items, folder, image_size):
    for class_name in CLASSES:
        os.makedirs(os.path.join(folder, class_name), exist_ok=True)
    for s in items:
        name = os.path.splitext(os.path.basename(s["file"]))[0].replace("%", "_")
        save_image(
            s["file"], os.path.join(folder, s["label"], f"{name}.png"), image_size
        )


def write_dataset_description(path, data_path, name, tags, description):
    with open(path, "w") as f:
        json.dump(
            {
                "name": name,
                "path": data_path,
                "data_type": "images",
                # The CLI splits `tags` on commas, it must be a string
                "tags": tags,
                "description": description,
            },
            f,
            indent=2,
        )


def counts(items):
    return {c: sum(s["label"] == c for s in items) for c in CLASSES}


if __name__ == "__main__":
    args = parse_args()
    root_folder = os.path.abspath(os.path.expanduser(args.root_folder))
    if not os.path.isdir(root_folder):
        raise SystemExit(f"Error: root folder does not exist: '{root_folder}'")

    print("--- Step 1: Downloading and extracting COVID-CT ---")
    raw_folder = download_and_extract(root_folder, args.base_url)

    samples = list_samples(raw_folder)
    print(f"Found {len(samples)} slices: {counts(samples)}")

    print("\n--- Step 2: Non-IID split into two hospitals ---")
    rng = np.random.default_rng(args.seed)
    node_of_group = assign_nodes(samples, args.covid_share, args.noncovid_mix, rng)

    table = []
    for node in NODES:
        component_folder = os.path.join(os.getcwd(), node)
        data_folder = os.path.join(component_folder, "data")

        if node_component.is_component_existing(component_folder) and not args.force:
            print(
                f"Warning: component '{component_folder}' already exists. "
                "Skipping node initialisation. Use --force to overwrite."
            )
        else:
            node_component.initiate(component_folder)

        if os.path.isdir(data_folder):
            shutil.rmtree(data_folder)

        node_samples = [s for s in samples if node_of_group[s["group"]] == node]
        train, test = split_holdout(node_samples, args.holdout, args.seed)
        for split_name, items in (("train", train), ("holdout", test)):
            split_folder = os.path.join(data_folder, split_name)
            write_split(items, split_folder, args.image_size)
            tag = "covid-ct" if split_name == "train" else "covid-ct-holdout"
            write_dataset_description(
                os.path.join(component_folder, f"dataset_{split_name}.json"),
                split_folder,
                name=f"covid-ct-{split_name}",
                tags=tag,
                description=f"COVID-CT {split_name} split of {node}",
            )
            c = counts(items)
            sources = sorted({s["source"] for s in items})
            table.append((node, split_name, c["CT_COVID"], c["CT_NonCOVID"], sources))

    print(f"\n{'node':<12}{'split':<9}{'COVID':>6}{'NonCOVID':>10}  sources")
    for node, split_name, n_covid, n_noncovid, sources in table:
        print(
            f"{node:<12}{split_name:<9}{n_covid:>6}{n_noncovid:>10}  {', '.join(sources)}"
        )

    print("\n--- Setup complete ---")
    print(
        "\nTo add the train data to each node (image-folder dataset, tag 'covid-ct'):"
    )
    for node in NODES:
        print(
            f"\tfedbiomed node -p {node} dataset add --file {node}/dataset_train.json"
        )
    print("\nTo start each node:")
    for node in NODES:
        print(f"\tfedbiomed node -p {node} start")
