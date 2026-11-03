"""Fake Nipoppy reader for NipoppyController / NipoppyDataset tests"""

import pandas as pd

PARTICIPANT = "participant_id"
SESSION = "session_id"

# One derivative, as (pipeline, version, path), exposed as a single column
DERIVATIVE = ("freesurfer", "7.3.2", "idp/fs_stats/thickness.tsv")
DERIVATIVE_COLUMN = "thickness"


def make_nipoppy_phenotypes() -> pd.DataFrame:
    """Phenotypes of a small Nipoppy study.

    - sub-01, sub-02, sub-03 have a `baseline` session
    - sub-01 also has a `followup` session
    - sub-03 has a missing `age` at baseline
    """
    index = pd.MultiIndex.from_tuples(
        [
            ("sub-01", "baseline"),
            ("sub-01", "followup"),
            ("sub-02", "baseline"),
            ("sub-03", "baseline"),
        ],
        names=[PARTICIPANT, SESSION],
    )
    return pd.DataFrame(
        {
            "age": [70.0, 72.0, 65.0, None],
            "sex": [0.0, 0.0, 1.0, 1.0],
            "diagnosis": [1, 1, 0, 0],
        },
        index=index,
    )


class FakeNipoppyReader:
    """Stand-in for `NipoppyReader`, backed by an in-memory dataframe."""

    # Same structure as `nipoppy.study.Study.get_installed_pipelines`
    installed_pipelines = {
        "bidsification": {},
        "processing": {"freesurfer": ["7.3.2"]},
        "extraction": {"fs_stats": ["0.2.1"]},
    }

    def __init__(self, path):
        self.path = path
        self.phenotypes = make_nipoppy_phenotypes()
        self.read_calls = []

    def read(self, phenotypes=None, derivatives=None):
        self.read_calls.append({"phenotypes": phenotypes, "derivatives": derivatives})
        if phenotypes is None and derivatives is None:
            return self.phenotypes.copy()
        df = self.phenotypes[list(phenotypes or [])].copy()
        for derivative in derivatives or []:
            assert tuple(derivative) == DERIVATIVE, f"unknown derivative {derivative}"
            df[DERIVATIVE_COLUMN] = [2.5, 2.6, 2.4, 2.3]
        return df

    def get_schema(self):
        return {str(c): str(t) for c, t in self.phenotypes.dtypes.items()}

    def get_installed_pipelines(self):
        return self.installed_pipelines

    def __len__(self):
        # number of participant-session rows, as `nipoppy.study.Study.__len__`
        return len(self.phenotypes)
