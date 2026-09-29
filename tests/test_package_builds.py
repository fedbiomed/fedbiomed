"""Check distribution boundaries and rebuilding outside the repository.

Run with pytest and hatchling installed, using --noconftest -c /dev/null to
avoid runtime fixtures. No runtime dependencies or Yarn are needed.
"""

import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("hatchling")

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "package", ["fedbiomed", "fedbiomed_node_api", "fedbiomed_gui"]
)
def test_distribution_contents_and_sdist_rebuild(package, tmp_path):
    source = ROOT if package == "fedbiomed" else ROOT / package
    staged = tmp_path / "source"
    staged.mkdir()
    if package == "fedbiomed":
        paths = ["fedbiomed", "docs/tutorials", "envs/common", "notebooks"]
    else:
        paths = [p.name for p in source.iterdir() if p.name != "ui"]
    for name in set(paths + ["pyproject.toml", "README.md", "LICENSE.md"]):
        origin = source / name
        target = staged / name
        if origin.is_dir():
            shutil.copytree(
                origin,
                target,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "dist"),
            )
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(origin, target)

    if package == "fedbiomed":
        # Model the monorepo: sibling packages must not leak into core artifacts.
        for sibling in ("fedbiomed_node_api", "fedbiomed_gui"):
            (staged / sibling).mkdir()
            (staged / sibling / "__init__.py").write_text("# excluded sibling\n")

    env = dict(os.environ)
    env.pop("FBM_SKIP_FRONTEND_BUILD", None)
    # Core/API must build with no frontend tooling, without the escape hatch.
    env["PATH"] = str(tmp_path / "no-tools")
    if package == "fedbiomed_gui":
        env["FBM_SKIP_FRONTEND_BUILD"] = "1"
        assets = staged / "ui" / "build"
        assets.mkdir(parents=True)
        (assets / "index.html").write_text("<html>packaging fixture</html>")
        (assets / "main.js").write_text("window.fixture = true;")
        (staged / "ui" / "node_modules").mkdir()
        (staged / "ui" / "node_modules" / "excluded.js").write_text("excluded")
        for name in ("package.json", "yarn.lock", "src", "public"):
            origin = source / "ui" / name
            target = staged / "ui" / name
            if origin.is_dir():
                shutil.copytree(origin, target)
            else:
                shutil.copyfile(origin, target)

    def build(directory, target):
        result = subprocess.run(
            [sys.executable, "-m", "hatchling", "build", "-t", target],
            cwd=directory,
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    def check_wheel(wheel):
        with zipfile.ZipFile(wheel) as archive:
            names = set(archive.namelist())
            assert f"{package}/__init__.py" in names
            for other in {"fedbiomed", "fedbiomed_node_api", "fedbiomed_gui"} - {
                package
            }:
                assert not any(name.startswith(other + "/") for name in names)
            assert not any(
                "node_modules/" in name or "__pycache__/" in name for name in names
            )
            if package == "fedbiomed_node_api":
                assert f"{package}/config_gui.ini" in names
                assert f"{package}/routes/authentication.py" in names
            elif package == "fedbiomed_gui":
                assert f"{package}/server/wsgi.py" in names
                assert f"{package}/ui/build/index.html" in names
                assert f"{package}/ui/build/main.js" in names
                assert not any("/ui/src/" in name for name in names)
                assert f"{package}/hatch_build.py" not in names
            else:
                assert any("/share/fedbiomed/notebooks/" in name for name in names)

    build(staged, "wheel")
    check_wheel(next((staged / "dist").glob("*.whl")))
    build(staged, "sdist")
    extracted = tmp_path / "extracted"
    with tarfile.open(next((staged / "dist").glob("*.tar.gz"))) as archive:
        names = archive.getnames()
        assert not any("node_modules/" in name for name in names)
        if package == "fedbiomed":
            assert not any(
                "/fedbiomed_gui/" in name or "/fedbiomed_node_api/" in name
                for name in names
            )
            assert not any(name.endswith("/hatch_build.py") for name in names)
        archive.extractall(extracted, filter="data")
    rebuilt = next(extracted.iterdir())
    build(rebuilt, "wheel")
    check_wheel(next((rebuilt / "dist").glob("*.whl")))
