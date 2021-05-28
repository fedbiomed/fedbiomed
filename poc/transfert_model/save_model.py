"""
poc on how to transfer a full NN torch model between to processes
"""


import inspect
#
# declare a local model
#
import torch
import torch.nn as nn
import torch.optim as optim

class Net(nn.Module):
    """
    model definition
    """
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


#
# validate the code
#
if __name__ == "__main__":

    # declare the model
    net = Net()
    print(net)

    print("=" * 60)
    for p in net.parameters():
        print(type(p.data), p.size())


    # initialize the model
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    print("=" * 60)
    for p in net.parameters():
        print(type(p.data), p.size())

    # save class in .py file
    s  = "import torch\n"
    s += "import torch.nn as nn\n"
    s += "\n"
    s += inspect.getsource(Net)

    file = open("my_model.py", "w")
    print(s)
    print(s.__class__)
    file.write(s)
    file.close()

    # save state with torch
    torch.save(net.state_dict(), "my_model.pt")
