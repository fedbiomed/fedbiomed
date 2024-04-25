from fedbiomed.common.training_plans import FedPerceptron
from fedbiomed.common.data import DataManager


class SkLearnClassifierTrainingPlan(FedPerceptron):
    def init_dependencies(self):
        """Define additional dependencies.
        return ["from torchvision import datasets, transforms",
                "from torch.utils.data import DataLoader"]

    def training_data(self):
        
        In this case, we rely on torchvision functions for preprocessing the images.
        """
        return ["from torchvision import datasets, transforms",]

    def training_data(self):
        """Prepare data for training.
        
        This function loads a MNIST dataset from the node's filesystem, applies some
        preprocessing and converts the full dataset to a numpy array. 
        Finally, it returns a DataManager created with these numpy arrays.
        """
        transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
        dataset = datasets.MNIST(self.dataset_path, train=True, download=False, transform=transform)
        
        X_train = dataset.data.numpy()
        X_train = X_train.reshape(-1, 28*28)
        Y_train = dataset.targets.numpy()
        return DataManager(dataset=X_train, target=Y_train,  shuffle=False)
