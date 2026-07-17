import atexit
import os
import shutil
import tempfile

# Default researcher workflow tests to debug-mode exception behavior.
os.environ.setdefault("FBM_DEBUG", "1")

# Redirect the researcher component created on `fedbiomed.researcher.config`
# import to a temp dir, so tests never write it into the repository.
if "FBM_RESEARCHER_COMPONENT_ROOT" not in os.environ:
    _researcher_root = tempfile.mkdtemp(prefix="fbm-researcher-test-")
    os.environ["FBM_RESEARCHER_COMPONENT_ROOT"] = _researcher_root
    atexit.register(shutil.rmtree, _researcher_root, ignore_errors=True)
