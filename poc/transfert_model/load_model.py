#!/usr/bin/env python
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
    try:
        sys.path.insert(0, ".")
        from my_model import Net
        sys.path.pop(0)

        # declare the model
        net = Net()
        print(net)
    except:
        e = sys.exc_info()
        print("Cannot load model: ", e)
        sys.exit(-1)

    net.load_state_dict(torch.load("my_model.pt"))
    net.eval()
