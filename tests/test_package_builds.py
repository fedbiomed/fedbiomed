"""Check distribution boundaries and rebuilding outside the repository.

Run with pytest and hatchling installed, using --noconftest -c /dev/null to
avoid runtime fixtures. No runtime dependencies or Yarn are needed.
"""

import ast
import configparser
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path

import pytest
import tomllib
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

pytest.importorskip("hatchling")

ROOT = Path(__file__).resolve().parents[1]


def core_version():
    module = ast.parse((ROOT / "fedbiomed" / "__init__.py").read_text())
    for node in module.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        ):
            return Version(ast.literal_eval(node.value))
    raise AssertionError("Core version is missing")


def test_coordinated_versions_and_pins():
    version = core_version()
    core = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    for folder, dependency in (
        ("fedbiomed_node_api", "fedbiomed"),
        ("fedbiomed_gui", "fedbiomed-node-api"),
    ):
        project = tomllib.loads((ROOT / folder / "pyproject.toml").read_text())[
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
    # All distributions must build without frontend tools or a skip-build flag.
    env["PATH"] = str(tmp_path / "no-tools")
    if package == "fedbiomed_gui":
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
            if package != "fedbiomed":
                # Wheels must ship the launcher module and register the executable
                # that the installer creates for the corresponding distribution.
                assert f"{package}/cli.py" in names
                entry_points = configparser.ConfigParser()
                entry_points.read_string(
                    archive.read(
                        next(
                            name
                            for name in names
                            if name.endswith(".dist-info/entry_points.txt")
                        )
                    ).decode()
                )
                assert (
                    entry_points["console_scripts"][package.replace("_", "-")]
                    == f"{package}.cli:run"
                )
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
    if package == "fedbiomed_gui":
        for asset in ("index.html", "main.js"):
            assert (rebuilt / "ui/build" / asset).read_bytes() == (
                staged / "ui/build" / asset
            ).read_bytes()
    build(rebuilt, "wheel")
    check_wheel(next((rebuilt / "dist").glob("*.whl")))

    if package != "fedbiomed":
        if package == "fedbiomed_gui":
            # Editable checkouts can install before building the frontend.
            shutil.rmtree(staged / "ui/build")
        editable = tmp_path / "editable"
        editable.mkdir()
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; from hatchling.build import build_editable; build_editable(sys.argv[1])",
                str(editable),
            ],
            cwd=staged,
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        with zipfile.ZipFile(next(editable.glob("*.whl"))) as archive:
            paths = [name for name in archive.namelist() if name.endswith(".pth")]
            assert len(paths) == 1
            assert Path(archive.read(paths[0]).decode().strip()) == staged.parent


@pytest.mark.parametrize("target", ["wheel", "sdist"])
@pytest.mark.parametrize("index_only", [False, True])
def test_gui_distribution_rejects_missing_assets(tmp_path, target, index_only):
    for name in (
        "pyproject.toml",
        "hatch_build.py",
        "README.md",
        "LICENSE.md",
        "__init__.py",
        "cli.py",
    ):
        shutil.copyfile(ROOT / "fedbiomed_gui" / name, tmp_path / name)
    if index_only:
        assets = tmp_path / "ui/build"
        assets.mkdir(parents=True)
        (assets / "index.html").write_text("<html>Incomplete bundle</html>")
    result = subprocess.run(
        [sys.executable, "-m", "hatchling", "build", "-t", target],
        cwd=tmp_path,
        # The former escape hatch must not permit broken release artifacts.
        env={
            **os.environ,
            "PATH": str(tmp_path / "no-tools"),
            "FBM_SKIP_FRONTEND_BUILD": "1",
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "frontend assets are missing or incomplete" in result.stderr
    assert "yarn install --frozen-lockfile" in result.stderr
