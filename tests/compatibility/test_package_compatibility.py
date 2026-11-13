"""
Verify Fed-BioMed distribution builds, wheel contents and installations.

Run from the repository root with pytest and hatchling available.
Runtime dependencies and Yarn are not needed: each package is checked in isolation.
The GUI uses a small prebuilt frontend fixture instead of invoking Yarn.
"""

import ast
import os
import shutil
import subprocess
import sys
import tarfile
import tomllib
import venv
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
    module = ast.parse(
        (PROJECTS["fedbiomed"] / "src/fedbiomed/__init__.py").read_text()
    )
    for node in module.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        ):
            return Version(ast.literal_eval(node.value))
    raise AssertionError("Core version is missing")


def run_command(args, *, cwd, env=None):
    result = subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"Command failed: {args!r}\n{result.stdout}\n{result.stderr}"
    )
    return result


def build(directory, target, env):
    run_command(
        [sys.executable, "-m", "hatchling", "build", "-t", target],
        cwd=directory,
        env=env,
    )


def build_env(package, workspace):
    env = os.environ.copy()
    env.pop("FBM_SKIP_FRONTEND_BUILD", None)
    # Check that core/API builds do not implicitly depend on Yarn.
    env["PATH"] = str(workspace / "no-frontend-tools")
    if package == "fedbiomed_gui":
        env["FBM_SKIP_FRONTEND_BUILD"] = "1"
    return env


def create_venv(path):
    venv.EnvBuilder(with_pip=True).create(path)
    return path / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def isolated_env():
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    return env


def wheel_metadata(archive):
    metadata_name = next(
        name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
    )
    return BytesParser().parsebytes(archive.read(metadata_name))


def check_wheel_metadata_and_contents(package, wheel):
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        metadata = wheel_metadata(archive)
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
                not (web | {"fedbiomed-gui", "fedbiomed-node-api"}) & selected().keys()
            )
            for extra, dependency in (
                ("node-api", "fedbiomed-node-api"),
                ("gui", "fedbiomed-gui"),
            ):
                assert (
                    str(selected(extra)[dependency].specifier) == f"=={core_version()}"
                )
        elif package == "fedbiomed_node_api":
            assert selected().keys() == web | {"fedbiomed"}
            assert str(selected()["fedbiomed"].specifier) == f"=={core_version()}"
        else:
            assert set(selected()) == {"fedbiomed-node-api"}
            assert (
                str(selected()["fedbiomed-node-api"].specifier) == f"=={core_version()}"
            )

        assert f"{package}/__init__.py" in names
        for other in set(PROJECTS) - {package}:
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
            assert "fedbiomed/common/__init__.py" in names
            assert "fedbiomed/src/__init__.py" not in names
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


def check_installed_package(python, package, cwd, *, editable=False, source=None):
    code = [
        "from pathlib import Path",
        "import importlib, sysconfig",
        f"package = importlib.import_module({package!r})",
        "assert Path(package.__file__).is_file()",
    ]
    if editable:
        code.append(
            f"assert Path(package.__file__).resolve() == "
            f"Path({str(source / 'src' / package / '__init__.py')!r}).resolve()"
        )
    else:
        code.append("assert 'site-packages' in Path(package.__file__).parts")

    if package == "fedbiomed":
        code += [
            "assert package.__version__",
            "import fedbiomed.common",
            "assert Path(fedbiomed.common.__file__).is_file()",
        ]
        if not editable:
            code += [
                "shared = Path(sysconfig.get_path('data')) / 'share/fedbiomed'",
                "assert (shared / 'envs/common/default_biprimes/biprime0.json').is_file()",
                "assert (shared / 'envs/common/default_training_plans').is_dir()",
                "assert (shared / 'notebooks').is_dir()",
            ]
    elif package == "fedbiomed_node_api":
        code.append(
            "assert Path(package.__file__).with_name('config_gui.ini').is_file()"
        )
    elif not editable:
        code.append(
            "assert (Path(package.__file__).parent / 'ui/build/index.html').is_file()"
        )

    run_command(
        [str(python), "-c", "\n".join(code)],
        cwd=cwd,
        env=isolated_env(),
    )


@pytest.fixture(scope="module", params=list(PROJECTS), ids=list(PROJECTS))
def staged_package(request, tmp_path_factory):
    """Copy one distribution into an independent, buildable temporary source tree."""
    package = request.param
    source = PROJECTS[package]
    workspace = tmp_path_factory.mktemp(f"packaging-{package}")
    staged = workspace / ("fedbiomed" if package == "fedbiomed" else "source")
    staged.mkdir()

    paths = ["src"]
    if package == "fedbiomed":
        paths += ["hatch_build.py", "docs/tutorials", "envs/common", "notebooks"]
    elif package == "fedbiomed_gui":
        paths.append("hatch_build.py")

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

    if package == "fedbiomed_gui":
        assets = staged / "ui/build"
        assets.mkdir(parents=True)
        (assets / "index.html").write_text("<html>packaging fixture</html>")
        (assets / "main.js").write_text("window.fixture = true;")
        modules = staged / "ui/node_modules"
        modules.mkdir()
        (modules / "excluded.js").write_text("excluded")
        for name in ("package.json", "yarn.lock", "src", "public"):
            origin = source / "ui" / name
            target = staged / "ui" / name
            if origin.is_dir():
                shutil.copytree(origin, target)
            else:
                shutil.copyfile(origin, target)

    return package, staged, workspace


@pytest.fixture(scope="module")
def built_wheel(staged_package):
    package, staged, workspace = staged_package
    build(staged, "wheel", build_env(package, workspace))
    (wheel,) = (staged / "dist").glob("*.whl")
    return wheel


def test_coordinated_versions_and_pins():
    """Versions and cross-distribution dependencies must be coordinated."""
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
        assert canonicalize_name(requirement.name) == dependency
        assert str(requirement.specifier) == f"=={version}"
        assert requirement.url is None


# -----------------------------------------------------------------------------
# 1. NORMAL WHEEL BUILD + SDIST REBUILD
# -----------------------------------------------------------------------------
def test_wheel_build_and_sdist_rebuild(staged_package, built_wheel):
    package, staged, workspace = staged_package
    check_wheel_metadata_and_contents(package, built_wheel)

    # Make sure the sdist can rebuild a valid wheel independently of the checkout.
    build(staged, "sdist", build_env(package, workspace))
    (sdist,) = (staged / "dist").glob("*.tar.gz")
    extracted = workspace / "extracted"
    extracted.mkdir()
    with tarfile.open(sdist) as archive:
        names = archive.getnames()
        assert not any("node_modules/" in name for name in names)
        if package == "fedbiomed":
            assert not any(
                "/fedbiomed_gui/" in name or "/fedbiomed_node_api/" in name
                for name in names
            )
            assert any(name.endswith("/hatch_build.py") for name in names)
        archive.extractall(extracted, filter="data")
    rebuilt = next(path for path in extracted.iterdir() if path.is_dir())
    if package == "fedbiomed_gui":
        # The sdist intentionally has frontend sources, not compiled assets.
        # Supply a minimal prebuilt frontend fixture for an offline rebuild.
        assets = rebuilt / "ui/build"
        assets.mkdir(parents=True, exist_ok=True)
        (assets / "index.html").write_text("<html>packaging fixture</html>")
        (assets / "main.js").write_text("window.fixture = true;")
    build(rebuilt, "wheel", build_env(package, workspace))
    (rebuilt_wheel,) = (rebuilt / "dist").glob("*.whl")
    check_wheel_metadata_and_contents(package, rebuilt_wheel)


# -----------------------------------------------------------------------------
# 2. INSTALL A REAL WHEEL INTO A CLEAN VENV AND IMPORT IT
# -----------------------------------------------------------------------------
def test_wheel_install(staged_package, built_wheel):
    package, _staged, workspace = staged_package
    python = create_venv(workspace / "wheel-venv")
    run_command(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--disable-pip-version-check",
            str(built_wheel),
        ],
        cwd=workspace,
        env=isolated_env(),
    )
    check_installed_package(python, package, workspace)


# -----------------------------------------------------------------------------
# 3. ALL PYTHON SOURCE FILES MUST MATCH THEIR CONTENTS IN THE WHEEL BYTE-FOR-BYTE
# -----------------------------------------------------------------------------
def test_wheel_matches_source(staged_package, built_wheel):
    package, staged, _workspace = staged_package
    source_root = staged / "src" / package
    # Source files whose bytes must be unchanged by packaging.
    source_files = {
        (Path(package) / path.relative_to(source_root)).as_posix(): path.read_bytes()
        for path in source_root.rglob("*")
        if path.is_file() and path.suffix in {".py", ".ini"}
    }
    assert source_files, "No Python source files found"
    with zipfile.ZipFile(built_wheel) as archive:
        wheel_files = {
            name: archive.read(name)
            for name in archive.namelist()
            if name.startswith(f"{package}/") and Path(name).suffix in {".py", ".ini"}
        }
    assert wheel_files.keys() == source_files.keys(), (
        f"Missing: {source_files.keys() - wheel_files.keys()}\n"
        f"Unexpected: {wheel_files.keys() - source_files.keys()}"
    )
    for name, content in source_files.items():
        assert wheel_files[name] == content, f"Wheel differs from source: {name}"


# -----------------------------------------------------------------------------
# 4. INSTALL AN EDITABLE WHEEL AND VERIFY EXISTING + NEW MODULES STAY EDITABLE
# -----------------------------------------------------------------------------
def test_editable_install(staged_package, tmp_path):
    package, staged, _workspace = staged_package
    source = tmp_path / "editable-source"
    shutil.copytree(
        staged,
        source,
        ignore=shutil.ignore_patterns("dist", "__pycache__", "*.pyc", "node_modules"),
    )

    python = create_venv(tmp_path / "venv")
    env = isolated_env()

    if package == "fedbiomed_gui":
        env["FBM_SKIP_FRONTEND_BUILD"] = "1"
    run_command(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--disable-pip-version-check",
            "-e",
            str(source),
        ],
        cwd=tmp_path,
        env=env,
    )
    check_installed_package(python, package, tmp_path, editable=True, source=source)

    package_root = source / "src" / package
    with (package_root / "__init__.py").open("a") as stream:
        stream.write("\n_editable_probe = 'updated'\n")
    (package_root / "_editable_probe.py").write_text("VALUE = 'new module'\n")
    run_command(
        [
            str(python),
            "-c",
            "from pathlib import Path\n"
            f"import {package} as package\n"
            "assert package._editable_probe == 'updated'\n"
            f"from {package}._editable_probe import VALUE\n"
            "assert VALUE == 'new module'\n"
            f"assert Path(package.__file__).resolve() == "
            f"Path({str(package_root / '__init__.py')!r}).resolve()\n",
        ],
        cwd=tmp_path,
        env=isolated_env(),
    )
