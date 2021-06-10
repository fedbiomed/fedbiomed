#!/usr/bin/env python
#
# test multipleheritance with fedbiomed and torch
#


import torch
import torch.nn as nn

from   fedbiomed.common.torchnn import Torchnn

class Net(Torchnn):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def training_step(self, data, target):
        output = self.forward(data)
        loss   = torch.nn.functional.nll_loss(output, target)
        return loss


#
# validate the code
#
if __name__ == "__main__":

    # declare the model
    net = Net()

    # debug
    for p in net.parameters():
        print(type(p.data), p.size())

    # save class in .py file
    #net.add_dependency("import torch")
    #net.add_dependency("import torch.nn as nn")
    net.save_code("my_model.py")

    Net.save_code("prout.py")

    # save state with torch
    net.save("my_model.pt")
