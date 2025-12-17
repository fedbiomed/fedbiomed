#!/usr/bin/env python

from pathlib import Path

from nipoppy import NipoppyDataRetriever

dataset_root = Path(__file__).parent / "my_dataset"

api = NipoppyDataRetriever(dataset_root)

df = api.get_tabular_data(
    phenotypes=[
        "nb:Age",
        "nb:Sex",
        "nb:Diagnosis",
        "snomed:859351000000102",  # MoCA
    ],
    derivatives=[
        ("freesurfer", "7.3.2", "idp/fs_stats-0.2.1/fs7.3.2-aseg-volume.tsv"),
        (
            "freesurfer",
            "7.3.2",
            "idp/fs_stats-0.2.1/fs7.3.2-aparc-thickness.tsv",
        ),
    ],
)

print(df)
print(df.iloc[0]["nb:Age", "Left-Lateral-Ventricle"])
print(df.columns)
