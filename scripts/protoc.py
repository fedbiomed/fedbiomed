import os

import grpc_tools
from grpc_tools import protoc

# gRPC google proto files
GRPC_PATH = grpc_tools.__path__[0]

# Base module path
BASE_DIR = os.path.normpath(
                    os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
            )

INPUT_PATH = os.path.join(f"{BASE_DIR}", 'fedbiomed', 'proto')
OUTPUT_PATH = INPUT_PATH



def compile_proto_files() -> None:
    """Compiles proto files in to fedbiomed/proto directory"""

    cmd = [
        "grpc_tools.protoc",
        f"--proto_path={GRPC_PATH}/_proto",
        f"--proto_path={INPUT_PATH}",
        # Output path
        f"--python_out=.",
        f"--grpc_python_out=.",
        f"--pyi_out=.",
        # Proto files should be kept in folder respecting fedbiomed proto module folder structure.
        # Otherwise, proto buffers ends up with broken imports within the fedbiomed module.
        f"{INPUT_PATH}/fedbiomed/**/*.proto"
    ]

    status = protoc.main(cmd)

    if status != 0:
        raise Exception(f"Error while compiling proto files: exit " \
                        "with status code {status}. Command: {cmd}")


if __name__ == "__main__":
    compile_proto_files()