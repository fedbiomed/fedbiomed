#!/usr/bin/env python3
"""Install released/candidate Core and run the end-to-end test in both directions."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN_DIR = ROOT / "tests/compatibility"


def run_pair(node_python, researcher_python):
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tests/end2end/e2e_node_researcher_compatibility.py"),
            "--node-python",
            str(node_python),
            "--researcher-python",
            str(researcher_python),
        ],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-python", type=Path)
    parser.add_argument("--researcher-python", type=Path)
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    parser.add_argument(
        "--baseline", type=Path, default=PLAN_DIR / "supported_versions.json"
    )
    args = parser.parse_args()
    if args.node_python or args.researcher_python:
        if not (args.node_python and args.researcher_python):
            parser.error("Provide both --node-python and --researcher-python")
        run_pair(args.node_python.absolute(), args.researcher_python.absolute())
    else:
        from setup_and_check_core_api_compatibility import (
            check_candidate_wheels,
            installed_environment,
        )

        wheels, _ = check_candidate_wheels(args.dist)
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))["baselines"][
            "core"
        ]
        with installed_environment([f"fedbiomed[researcher]=={baseline}"]) as (
            old,
            _,
            _,
        ):
            candidate = f"fedbiomed[researcher] @ {wheels['core'].as_uri()}"
            with installed_environment([candidate]) as (new, _, _):
                print("Released node + candidate researcher", flush=True)
                run_pair(old, new)
                print("Candidate node + released researcher", flush=True)
                run_pair(new, old)


if __name__ == "__main__":
    main()
