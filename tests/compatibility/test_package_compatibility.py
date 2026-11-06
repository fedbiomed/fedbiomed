"""Check distribution boundaries and rebuilding outside the repository.

Run with pytest and hatchling installed. No runtime dependencies or Yarn are
needed; these checks are independent of package runtime fixtures.
"""

import ast
import os
import shutil
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

pytest.importorskip("hatchling")

ROOT = Path(__file__).resolve().parents[2]
PROJECTS = {
    "fedbiomed": ROOT / "packages/fedbiomed",
    "fedbiomed_node_api": ROOT / "packages/node-api",
    "fedbiomed_gui": ROOT / "packages/node-ui",
}


def core_version():
    module = ast.parse((PROJECTS["fedbiomed"] / "src/__init__.py").read_text())
    for node in module.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        ):
            return Version(ast.literal_eval(node.value))
    raise AssertionError("Core version is missing")


def test_coordinated_versions_and_pins():
    version = core_version()
    core = tomllib.loads((PROJECTS["fedbiomed"] / "pyproject.toml").read_text())[
        "project"
    ]
    for folder, dependency in (
        ("fedbiomed_node_api", "fedbiomed"),
        ("fedbiomed_gui", "fedbiomed-node-api"),
    ):
        project = tomllib.loads((PROJECTS[folder] / "pyproject.toml").read_text())[
            "project"
        ]
        assert Version(project["version"]) == version
        requirement = next(
            Requirement(value)
            for value in project["dependencies"]
            if canonicalize_name(Requirement(value).name) == dependency
        )
        assert str(requirement.specifier) == f"=={version}"
        assert requirement.url is None
    for extra, dependency in (
        ("node-api", "fedbiomed-node-api"),
        ("gui", "fedbiomed-gui"),
    ):
        requirements = core["optional-dependencies"][extra]
        assert len(requirements) == 1
        requirement = Requirement(requirements[0])
        assert requirement.name == dependency
        assert str(requirement.specifier) == f"=={version}"
        assert requirement.url is None


@pytest.mark.parametrize(
    "package", ["fedbiomed", "fedbiomed_node_api", "fedbiomed_gui"]
)
def test_distribution_contents_and_sdist_rebuild(package, tmp_path):
    source = PROJECTS[package]
    staged = tmp_path / "source"
    staged.mkdir()
    if package == "fedbiomed":
        paths = ["src", "hatch_build.py", "docs/tutorials", "envs/common", "notebooks"]
    elif package == "fedbiomed_node_api":
        paths = ["src", "config_gui.ini"]
    else:
        paths = ["__init__.py", "server", "hatch_build.py"]
    for name in set(paths + ["pyproject.toml", "README.md", "LICENSE.md"]):
        origin = source / name
        if package == "fedbiomed" and name in {
            "README.md",
            "docs/tutorials",
            "envs/common",
            "notebooks",
        }:
            origin = ROOT / name
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
            metadata = BytesParser().parsebytes(
                archive.read(
                    next(name for name in names if name.endswith(".dist-info/METADATA"))
                )
            )
            assert Version(metadata["Version"]) == core_version()
            requirements = [
                Requirement(value) for value in metadata.get_all("Requires-Dist", [])
            ]
            assert all(requirement.url is None for requirement in requirements)

            def selected(extra=""):
                return {
                    canonicalize_name(requirement.name): requirement
                    for requirement in requirements
                    if requirement.marker is None
                    or requirement.marker.evaluate({"extra": extra})
                }

            web = {"flask", "flask-jwt-extended", "jsonschema", "cachelib", "gunicorn"}
            if package == "fedbiomed":
                assert (
                    not (web | {"fedbiomed-gui", "fedbiomed-node-api"})
                    & selected().keys()
                )
                for extra, dependency in (
                    ("node-api", "fedbiomed-node-api"),
                    ("gui", "fedbiomed-gui"),
                ):
                    assert (
                        str(selected(extra)[dependency].specifier)
                        == f"=={core_version()}"
                    )
            elif package == "fedbiomed_node_api":
                assert selected().keys() == web | {"fedbiomed"}
                assert str(selected()["fedbiomed"].specifier) == f"=={core_version()}"
            else:
                assert set(selected()) == {"fedbiomed-node-api"}
                assert (
                    str(selected()["fedbiomed-node-api"].specifier)
                    == f"=={core_version()}"
                )
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
                assert any(
                    name.endswith(
                        "/share/fedbiomed/envs/common/default_biprimes/biprime0.json"
                    )
                    for name in names
                )
                assert any(
                    "/share/fedbiomed/envs/common/default_training_plans/" in name
                    for name in names
                )
                assert metadata.get_payload() == (ROOT / "README.md").read_text()

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
            assert any(name.endswith("/hatch_build.py") for name in names)
        archive.extractall(extracted, filter="data")
    rebuilt = next(extracted.iterdir())
    build(rebuilt, "wheel")
    check_wheel(next((rebuilt / "dist").glob("*.whl")))

    # These prefix-renamed source layouts use normal wheel installs, not editable
    # installs. Confirm the import name works without the repository on sys.path.
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import importlib, sys; sys.path.insert(0, sys.argv[1]); "
            "module = importlib.import_module(sys.argv[2]); "
            "assert module.__file__.startswith(sys.argv[1])",
            str(next((rebuilt / "dist").glob("*.whl"))),
            package,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
