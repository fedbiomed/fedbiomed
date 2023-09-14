eval "$(conda shell.$(basename $SHELL) activate fedbiomed-node)"

./scripts/fedbiomed_run node list | grep MNIST
if [ $? -eq 0 ]
then
    echo "MNIST already installed, not doing anything"
else
    echo "MNIST seems not to be installed, installing it"
    mkdir -p $HOME/Data/fedbiomed # \todo brittle workaround for bug #862
    ./scripts/fedbiomed_run node --add-mni
fi


