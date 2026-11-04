import subprocess
import sys

import pytest

from fedbiomed.common.dataset_reader._nipoppy_reader import NipoppyReader
from fedbiomed.common.exceptions import FedbiomedError


def test_importing_datasets_does_not_import_nipoppy():
    # Run in a new interpreter, since other tests may already have imported nipoppy.
    # Setting the module to `None` makes any attempt to import it raise `ImportError`.
    code = (
        "import sys\n"
        "sys.modules['nipoppy'] = None\n"
        "import fedbiomed.common.dataset\n"
        "from fedbiomed.common.dataset import NipoppyDataset\n"
        "NipoppyDataset(phenotypes=['nb:Age'])\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_reader_without_nipoppy_raises(tmp_path, mocker):
    mocker.patch.dict(sys.modules, {"nipoppy": None})

    with pytest.raises(FedbiomedError, match="pip install"):
        NipoppyReader(tmp_path)
