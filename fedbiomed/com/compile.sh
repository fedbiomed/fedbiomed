basedir=$(cd $(dirname $0) || exit ; pwd)
cd $basedir || exit

echo $basedir

python -m grpc_tools.protoc -I $basedir/service --python_out=$basedir/service --pyi_out=$basedir/service --grpc_python_out=$basedir/service $basedir/service/message.proto