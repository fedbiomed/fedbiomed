from grpc_tools import command


# TODO: Add this command to setup.py once it is created
def compile_proto_files() -> None: 
    """Builds all proto files existing in Fed-BioMed"""
    command.build_package_protos('.')


if __name__ == "__main__":
    compile_proto_files()