"""Validate the release set and rebuild its source archives without frontend tools."""

import argparse
import os
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from email.parser import BytesParser
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import (
    canonicalize_name,
    parse_sdist_filename,
    parse_wheel_filename,
)
from packaging.version import Version

PACKAGES = {"fedbiomed", "fedbiomed-node-api", "fedbiomed-gui"}


def verify(directory, expected_version=None):
    # Require one wheel and one source archive for each member of the release.
    wheels = list(directory.glob("*.whl"))
    sources = list(directory.glob("*.tar.gz"))
    assert len(wheels) == len(sources) == 3, (
        "Expected three wheels and three source archives"
    )
    wheel_versions = {}
    for wheel in wheels:
        name, version, _, tags = parse_wheel_filename(wheel.name)
        assert name not in wheel_versions, f"Duplicate wheel: {name}"
        assert {str(tag) for tag in tags} == {"py3-none-any"}
        wheel_versions[name] = version
        with zipfile.ZipFile(wheel) as archive:
            metadata = BytesParser().parsebytes(
                archive.read(
                    next(
                        path
                        for path in archive.namelist()
                        if path.endswith(".dist-info/METADATA")
                    )
                )
            )
            for value in metadata.get_all("Requires-Dist", []):
                # Development paths must not leak into published requirements;
                # sibling pins must match this coordinated release's version.
                requirement = Requirement(value)
                assert requirement.url is None, f"Local dependency leaked: {value}"
                if canonicalize_name(requirement.name) in PACKAGES:
                    assert str(requirement.specifier) == f"=={version}"
    source_versions = dict(parse_sdist_filename(path.name) for path in sources)
    assert set(wheel_versions) == PACKAGES
    assert source_versions == wheel_versions
    assert len(set(wheel_versions.values())) == 1, "Package versions differ"
    if expected_version is not None:
        # Normalize tags such as v6.4.1 using the same version rules as packaging.
        assert set(wheel_versions.values()) == {Version(expected_version)}, (
            f"Release tag {expected_version} does not match artifact versions"
        )

    with tempfile.TemporaryDirectory(prefix="fbm-sdist-check-") as tmp:
        root = Path(tmp)
        env = dict(os.environ, PATH=str(root / "no-frontend-tools"))
        # Use an absolute Python executable, but hide Node/Yarn and remove the
        # old escape hatch: source archives must genuinely be self-contained.
        env.pop("FBM_SKIP_FRONTEND_BUILD", None)
        for source in sources:
            unpacked = root / source.stem
            with tarfile.open(source) as archive:
                archive.extractall(unpacked, filter="data")
            (project,) = unpacked.iterdir()
            destination = root / (source.stem + "-wheel")
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "hatchling",
                    "build",
                    "-t",
                    "wheel",
                    "-d",
                    str(destination),
                ],
                cwd=project,
                env=env,
                check=True,
            )
            (rebuilt,) = destination.glob("*.whl")
            # Compare extracted contents, not ZIP bytes: compression/container
            # details are irrelevant, while missing or changed resources are not.
            original = directory / rebuilt.name
            with (
                zipfile.ZipFile(original) as expected,
                zipfile.ZipFile(rebuilt) as actual,
            ):
                assert set(expected.namelist()) == set(actual.namelist())
                for path in expected.namelist():
                    assert expected.read(path) == actual.read(path), (
                        f"Rebuilt content differs: {path}"
                    )
    print(
        "Three matching distributions verified; source rebuilds need no frontend tools."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--expected-version", help="Release tag to match, e.g. v6.4.1")
    args = parser.parse_args()
    verify(args.directory.resolve(), args.expected_version)
