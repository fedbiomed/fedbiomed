#!/usr/bin/env python

import glob
import os
import shutil

from grpc_tools import command


# TODO: Add this command to setup.py once it is created
def compile_proto_files() -> None:
    """Builds all proto files existing in Fed-BioMed"""
    command.build_package_protos(".")


if __name__ == "__main__":
    root_dir = os.path.abspath(
        os.path.join(
            __file__,
            "..",
            "..",
        )
    )
    proto_dir = os.path.abspath(
        os.path.join(__file__, "..", "..", "fedbiomed", "transport", "protocols")
    )
    print(f"Refreshing: {proto_dir} files")
    os.chdir(proto_dir)

    # removing old proto files
    for f in glob.glob("*.py"):
        print(f"- cleaning: {f}")
        os.remove(f)
    for f in glob.glob("*.pyi"):
        print(f"- cleaning: {f}")
        os.remove(f)
    shutil.rmtree("__pycache__", ignore_errors=True)

    # creating new ones
    os.chdir(root_dir)
    compile_proto_files()
    print()
    for f in glob.glob("*.py"):
        print(f"- creating: {f}")
