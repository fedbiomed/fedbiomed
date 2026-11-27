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


def test_get_sessions_and_schema_read_phenotypes_once(tmp_path, mocker):
    pytest.importorskip("nipoppy")
    from testsupport.fake_nipoppy_reader import make_nipoppy_phenotypes

    retriever = mocker.patch("nipoppy.NipoppyDataRetriever").return_value
    retriever.get_all_phenotypes.return_value = make_nipoppy_phenotypes()
    reader = NipoppyReader(tmp_path)

    assert reader.get_sessions() == ["baseline", "followup"]
    assert reader.get_schema() == {
        "age": "float64",
        "sex": "float64",
        "diagnosis": "int64",
    }
    retriever.get_all_phenotypes.assert_called_once()
