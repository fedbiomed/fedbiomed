# `NipoppyDataRetriever` example

Install the latest development version of `nipoppy` in a new or existing Python environment:

```bash
pip install git+https://github.com/nipoppy/nipoppy.git@main
```

The Nipoppy dataset is at `my_dataset`.
It contains dummy data from 2 subjects and 2 imaging sessions:
- `tabular/harmonized.tsv`: demographic/assessments ("phenotypic") data harmonized by the Neurobagel CLI
- `derivatives/freesurfer/7.3.2/idp/fs_stats-0.2.1/fs7.3.2-aparc-thickness.tsv`: FreeSurfer cortical thicknesses
- `derivatives/freesurfer/7.3.2/idp/fs_stats-0.2.1/fs7.3.2-aseg-volume.tsv`: FreeSurfer subcortical volumes

The script `nipoppy_data_retriever_example.py` shows how to instantiate a `NipoppyDataRetriever` object and use it to retrieve a `pandas.DataFrame` with harmonized phenotypic and imaging derivatives measures.
