#!/usr/bin/env python
"""
poc on how to read full NN torch model between to processes
"""


import sys

#
# validate the code
#
if __name__ == "__main__":

    importClass = 'Net'
    importModule = 'my_model'

    #
    try:
        sys.path.insert(0, ".")
        exec('import ' + importModule)
        sys.path.pop(0)
        trainClass = eval(importModule + '.' + importClass)
    except:
        e = sys.exc_info()
        print("Cannot import class ", importClass, " from module ", importModule, " - Error: ", e)
        sys.exit(-1)

    # declare the model
    try:
        net = trainClass()
        print(net)
    except:
        e = sys.exc_info()
        print("Cannot load model: ", e)
        sys.exit(-1)

    net.load("my_model.pt")
