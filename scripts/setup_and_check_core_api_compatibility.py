#!/usr/bin/env python3
"""Test candidate wheel(s) against explicitly supported published distributions.

Run from repository root after building all three wheels. Requires Python >=3.11.
No extra third-party Python libraries are needed for the test driver.

Supported release versions MUST be versions of the *split* distributions (not the
legacy monolithic fedbiomed release). Populate the baseline file after the
first split-package release. The script reports when historical coverage is
not yet possible instead of implying that such coverage passed.
"""

import argparse
import json
import os
import pathlib
import re
import subprocess
import tempfile
import venv
import zipfile
from contextlib import contextmanager
from email.parser import BytesParser

DIST = {"core": "fedbiomed", "api": "fedbiomed-node-api", "gui": "fedbiomed-gui"}
WHEEL_PREFIX = {
    "core": "fedbiomed-",
    "api": "fedbiomed_node_api-",
    "gui": "fedbiomed_gui-",
}
IMPORTS = {
    "core": "fedbiomed",
    "api": "fedbiomed_node_api.application",
    "gui": "fedbiomed_gui.cli",
}
CLIS = {"core": "fedbiomed", "api": "fedbiomed-node-api", "gui": "fedbiomed-ui"}
CONTRACT = (
    pathlib.Path(__file__).resolve().parents[1]
    / "tests/compatibility/historical_contract.py"
)


def wheel_info(path):
    with zipfile.ZipFile(path) as z:
        filename = next(n for n in z.namelist() if n.endswith(".dist-info/METADATA"))
        metadata = BytesParser().parsebytes(z.read(filename))
    return metadata["Name"].lower().replace("_", "-"), metadata["Version"]


def minimum_dependency_version(wheel_path, dependency):
    """Find our supported minimum for an internal distribution, if declared."""
    with zipfile.ZipFile(wheel_path) as archive:
        filename = next(
            n for n in archive.namelist() if n.endswith(".dist-info/METADATA")
        )
        metadata = BytesParser().parsebytes(archive.read(filename))
    for requirement in metadata.get_all("Requires-Dist", []):
        normalized = requirement.lower().replace("_", "-")
        if not re.match(rf"^{re.escape(dependency)}(?:\s|\(|$)", normalized):
            continue
        match = re.search(r">=\s*(\d+\.\d+\.\d+)", requirement)
        if match:
            return tuple(map(int, match.group(1).split(".")))
    return None


def version_tuple(value):
    return tuple(map(int, value.split(".")))


def call(args, *, env=None, cwd=None):
    print("$", " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), check=True, env=env, cwd=cwd)


def check_candidate_wheels(directory):
    wheels = {}
    versions = {}
    for key, prefix in WHEEL_PREFIX.items():
        hits = list(directory.glob(f"{prefix}*.whl"))
        if len(hits) != 1:
            raise ValueError(f"Expected one {key} wheel, got: {hits}")
        dist, version = wheel_info(hits[0])
        if dist != DIST[key]:
            raise ValueError(f"Unexpected wheel distribution: {hits[0]} -> {dist}")
        wheels[key], versions[key] = hits[0].resolve(), version
    return wheels, versions


def smoke_check(python, keys, *, env, cwd):
    # Tests import compatibility and CLI invocation, not the entire functional
    # node-to-researcher protocol. API contracts run separately below.
    imports = "; ".join(f"import {IMPORTS[key]}" for key in keys)
    call([python, "-c", imports], env=env, cwd=cwd)
    bin_dir = pathlib.Path(python).parent
    for key in keys:
        call([bin_dir / CLIS[key], "--help"], env=env, cwd=cwd)


@contextmanager
def installed_environment(requirements):
    with tempfile.TemporaryDirectory(prefix="fbm-backcompat-") as tmp:
        venv.EnvBuilder(with_pip=True).create(tmp)
        python = pathlib.Path(tmp) / (
            "Scripts/python.exe" if os.name == "nt" else "bin/python"
        )
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        call([python, "-m", "pip", "install", "--upgrade", "pip"], env=env, cwd=tmp)
        call([python, "-m", "pip", "install", *requirements], env=env, cwd=tmp)
        call([python, "-m", "pip", "check"], env=env, cwd=tmp)
        yield python, env, tmp


def verify_case(name, requirements, keys):
    print(f"\n=== BACKWARD COMPATIBILITY: {name} ===", flush=True)
    with installed_environment(requirements) as (python, env, tmp):
        smoke_check(python, keys, env=env, cwd=tmp)
        if "api" in keys:
            call(
                [python, CONTRACT, "fresh", pathlib.Path(tmp) / "node"],
                env=env,
                cwd=tmp,
            )


def verify_persisted_state(wheels, baselines):
    if not {"core", "api"} <= baselines.keys():
        print(
            "::warning::Persisted-state compatibility NOT TESTED: core and API baselines required."
        )
        return
    print(
        "\n=== BACKWARD COMPATIBILITY: released node files + candidate packages ===",
        flush=True,
    )
    with tempfile.TemporaryDirectory(prefix="fbm-historical-state-") as tmp:
        root = pathlib.Path(tmp) / "node"
        released = [f"{DIST[key]}=={baselines[key]}" for key in ("core", "api")]
        with installed_environment(released) as (python, env, cwd):
            call([python, CONTRACT, "seed", root], env=env, cwd=cwd)
        with installed_environment([str(wheel) for wheel in wheels.values()]) as (
            python,
            env,
            cwd,
        ):
            call([python, CONTRACT, "check", root], env=env, cwd=cwd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist", type=pathlib.Path, default=pathlib.Path("dist"))
    parser.add_argument(
        "--baseline",
        type=pathlib.Path,
        default=pathlib.Path("tests/compatibility/supported_versions.json"),
    )
    args = parser.parse_args()

    wheels, versions = check_candidate_wheels(args.dist)
    data = json.loads(args.baseline.read_text(encoding="utf-8"))
    baselines = data.get("baselines", {})
    for key, version in baselines.items():
        if key not in DIST or not isinstance(version, str) or not version:
            raise ValueError(f"Invalid baseline: {key}={version!r}")
        if version.split(".")[0] != versions[key].split(".")[0]:
            raise ValueError(
                f"Baseline {key}={version} has different major from candidate {versions[key]}"
            )

    # Check forward migration (a new consumer with old provider) only where
    # the candidate's declared lower bound permits that older provider.
    # pip's real resolver enforces whether combinations are supported.
    cases = []
    if "api" in baselines:
        cases.append(
            (
                "old API + candidate Core",
                [str(wheels["core"]), f"{DIST['api']}=={baselines['api']}"],
                ["core", "api"],
            )
        )
    if "gui" in baselines:
        cases.append(
            (
                "old GUI + candidate API + Core",
                [
                    str(wheels["core"]),
                    str(wheels["api"]),
                    f"{DIST['gui']}=={baselines['gui']}",
                ],
                ["core", "api", "gui"],
            )
        )
    if "core" in baselines:
        minimum = minimum_dependency_version(wheels["api"], DIST["core"])
        if minimum is None or version_tuple(baselines["core"]) >= minimum:
            cases.append(
                (
                    "candidate API + old Core",
                    [f"{DIST['core']}=={baselines['core']}", str(wheels["api"])],
                    ["core", "api"],
                )
            )
        else:
            print(
                f"Skipping unsupported old Core {baselines['core']}; candidate API requires >= {minimum}"
            )
    if "api" in baselines:
        minimum = minimum_dependency_version(wheels["gui"], DIST["api"])
        if minimum is None or version_tuple(baselines["api"]) >= minimum:
            cases.append(
                (
                    "candidate GUI + old API",
                    [
                        str(wheels["core"]),
                        f"{DIST['api']}=={baselines['api']}",
                        str(wheels["gui"]),
                    ],
                    ["core", "api", "gui"],
                )
            )
        else:
            print(
                f"Skipping unsupported old API {baselines['api']}; candidate GUI requires >= {minimum}"
            )

    if not cases:
        print(
            "::warning::No split-distribution historical baselines configured. Backward compatibility NOT TESTED; only candidate wheel tests apply."
        )
        return
    print("Candidate:", versions, "Historical baselines:", baselines, flush=True)
    for name, requirements, keys in cases:
        verify_case(name, requirements, keys)
    verify_persisted_state(wheels, baselines)
    print(f"Completed {len(cases)} historical compatibility smoke/API contract cases.")


if __name__ == "__main__":
    main()
