#!/usr/bin/env python
"""
poc on how to transfer a full NN torch model between to processes
"""


import inspect
#
# declare a local model
#
#import fedbiomed.torch.nn as nn  // SHOULD BE PROVIDED

# dependencies for the model
import torch
import torch.nn as nn

# dependencies fot the training_data routine // SHOULD BE provided by fedbiomed
from torchvision import datasets, transforms

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

        ############################################################
        # conventional design // can be hidden in the super() call
        self.optimizer = torch.optim.Adam(self.parameters(), lr = 1e-3)

        # data loading // should ne moved to another class
        self.batch_size = 100
        self.shuffle    = True

        # training // may be changed in training_routine ??
        self.device = "cpu"

        # list dependencies of the model
        self.dependencies = []

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    # provided by the researcher / can change
    def training_step(self, data, target):
        output = self.forward(data)
        loss   = torch.nn.functional.nll_loss(output, target)
        return loss

    #################################################
    # provided by fedbiomed
    def training_routine(self, epochs=2, log_interval = 10, dry_run=False):

        #use_cuda = torch.cuda.is_available()
        #device = torch.device("cuda" if use_cuda else "cpu")
        self.device = "cpu"

        for epoch in range(1, epochs + 1):

            training_data = self.training_data()
            for batch_idx, (data, target) in enumerate(training_data):
                self.train()
                data, target = data.to(device), target.to(device)
                self.optimizer().zero_grad()
                res = self.training_step(data, target)
                res.backward()
                self.optimizer.step()

                if batch_idx % log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch,
                        batch_idx * len(data),
                        len(training_data.dataset),
                        100 * batch_idx / len(training_data),
                        loss.item()))
                    #
                    # deal with the logger here
                    #

                    if dry_run:
                        return

    # provided by fedbiomed // necessary to save the model code into a file
    def add_dependency(self, dep):
        self.dependencies.append(dep)
        pass

    # provider by fedbiomed
    def save_code(self):

        content = ""
        for s in self.dependencies:
            content += s + "\n"

        content += "\n"
        content += inspect.getsource(Net)

        # try/except todo
        file = open("my_model.py", "w")
        file.write(content)
        file.close()

    # provided by fedbiomed
    def save(self, filename):
        return torch.save(self.state_dict(), filename)

    # provided by fedbiomed
    def load(self, filename):
        return self.load_state_dict(torch.load(filename))

    # provided by the fedbiomed / can be overloaded // need WORK
    def logger(self, msg, batch_index, log_interval = 10):
        pass

    # provided by the fedbiomed // should be moved in a DATA manipulation module
    def training_data(self, batch_size = 48, shuffle = True):

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset1 = datasets.MNIST('~/.data/', train=True, download=True,
                                  transform=transform)
        train_kwargs = {'batch_size': batch_size, 'shuffle': shuffle}
        data_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)

        return data_loader

#
# validate the code
#
if __name__ == "__main__":

    # declare the model
    net = Net()

    print(net)
    print("=" * 60)

    # initialize the model
    #optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    #print("=" * 60)
    #for p in net.parameters():
    #    print(type(p.data), p.size())

    # save class in .py file
    net.add_dependency("import torch")
    net.add_dependency("import torch.nn as nn")
    net.add_dependency("from torchvision import datasets, transforms") # remove
    net.save_code()

    # save state with torch
    #torch.save(net.state_dict(), "my_model.pt")
    net.save("my_model.pt")
