"""
poc on how to read full NN torch model between to processes
"""

import torch

import inspect
import sys

#
# validate the code
#
if __name__ == "__main__":

    #
    sys.path.insert(0, ".")
    from my_model import Net
    sys.path.pop(0)

    # declare the model
    net = Net()
    print(net)

    net.load_state_dict(torch.load("my_model.pt"))

    print("=" * 60)
    for p in net.parameters():
        print(type(p.data), p.size())

    net.eval()
