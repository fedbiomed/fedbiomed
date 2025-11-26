#!/usr/bin/env python

import glob
import os
import shutil
import subprocess
import sys


# TODO: Add this command to setup.py once it is created
def compile_proto_files(proto_dir: str) -> None:
    """Builds all proto files existing in Fed-BioMed transport protocols directory"""

    print(f"Building proto files in {proto_dir}...")

    # Find all .proto files in the protocols directory
    proto_files = glob.glob(os.path.join(proto_dir, "*.proto"))

    if not proto_files:
        print("No .proto files found in the protocols directory")
        return

    for proto_file in proto_files:
        print(f"Processing: {proto_file}")

        # Run protoc command for each proto file
        cmd = [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            f"--proto_path={proto_dir}",
            f"--python_out={proto_dir}",
            f"--pyi_out={proto_dir}",
            f"--grpc_python_out={proto_dir}",
            proto_file,
        ]

        try:
            subprocess.run(cmd, check=True, cwd=proto_dir)
            print(f"Successfully compiled: {os.path.basename(proto_file)}")

            # Fix the import in the generated gRPC file
            grpc_file = proto_file.replace(".proto", "_pb2_grpc.py")
            if os.path.exists(grpc_file):
                fix_grpc_imports(
                    grpc_file, os.path.basename(proto_file).replace(".proto", "_pb2")
                )

        except subprocess.CalledProcessError as e:
            print(f"Error compiling {proto_file}: {e}")
            raise


def fix_grpc_imports(grpc_file: str, pb2_module: str) -> None:
    """Fix the import statement in generated gRPC files to use relative imports"""

    with open(grpc_file, "r") as f:
        content = f.read()

    # Replace the absolute import with a relative import
    old_import = f"import {pb2_module} as {pb2_module.replace('_pb2', '__pb2')}"
    new_import = f"from . import {pb2_module} as {pb2_module.replace('_pb2', '__pb2')}"

    if old_import in content:
        content = content.replace(old_import, new_import)

        with open(grpc_file, "w") as f:
            f.write(content)

        print(f"Fixed imports in: {os.path.basename(grpc_file)}")


if __name__ == "__main__":
    proto_dir = os.path.abspath(
        os.path.join(__file__, "..", "..", "fedbiomed", "transport", "protocols")
    )
    print(f"Refreshing: {proto_dir} files")

    # removing old proto files
    for f in glob.glob(os.path.join(proto_dir, "*.py")):
        print(f"- cleaning: {f}")
        os.remove(f)
    for f in glob.glob(os.path.join(proto_dir, "*.pyi")):
        print(f"- cleaning: {f}")
        os.remove(f)

    pycache_path = os.path.join(proto_dir, "__pycache__")
    if os.path.exists(pycache_path):
        shutil.rmtree(pycache_path, ignore_errors=True)

    # creating new ones
    compile_proto_files(proto_dir)

    # List the newly created files
    print("\nGenerated files:")
    for f in glob.glob(os.path.join(proto_dir, "*.py")):
        print(f"- created: {os.path.basename(f)}")
    for f in glob.glob(os.path.join(proto_dir, "*.pyi")):
        print(f"- created: {os.path.basename(f)}")
