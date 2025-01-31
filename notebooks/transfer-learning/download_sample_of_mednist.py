import argparse
from pathlib import Path
import random
import shutil
import os
from typing import Any, Dict, List, Optional, Tuple, Union

from fedbiomed.common.utils import ROOT_DIR
from fedbiomed.common.data import DataLoadingPlan
from fedbiomed.common.exceptions import FedbiomedDatasetManagerError

from fedbiomed.node.dataset_manager import DatasetManager
from fedbiomed.node.config import NodeConfig
from fedbiomed.node.config import NodeComponent
from torchvision import transforms, datasets



def parse_args():
    """Argument Parser"""
    parser = argparse.ArgumentParser(description='MEDNIST sampler. Creates configuration files with sub-sets of the MedNIST dataset')
    parser.add_argument('-f', '--root_folder', required=False, type=str, default=os.getcwd(),
                        help='Folder where to save dataset')
    parser.add_argument('-F', '--force', action=argparse.BooleanOptionalAction, required=False, type=bool, default=False,
                        help='forces overwriting a config file')
    parser.add_argument('-n', '--number_nodes', required=False, type=int, default=1, help="number of nodes to create")
    parser.add_argument('-s', '--random_seed', required=False, type=int, default=None, help="random seed to make MedNIST dataset splitting reproducible")
    return parser.parse_args()


def manage_config_file(
    mednist_center_name: str = "mednist", **kwargs
):
    """Creates config file for subsampled datasets

    Args:
        args (argparse.ArgumentParser): args parser
        mednist_center_name (str, optional): name for the node id, as well as the config file and the databse json.
            Defaults to "mednist".
    """
    mednist_center_name = "node_" + mednist_center_name
    component = NodeComponent()
    center_folder = os.path.join(os.getcwd(), mednist_center_name)
    config = component.initiate(root=center_folder)

    return config

def ask_nb_sample_for_mednist_dataset() -> str:
    """Asks to the user the number of samples

    Returns:
        the value inputed by the user
    """
    ask_user = True
    while ask_user:
        val = input("How many samples do you want to load? (press enter for the full dataset)\n")
        if val == '':
            ask_user = False

        elif (int(val) < 58955 and int(val) > 5):
            ask_user = False

        else:
            print(f"Number of samples exceed size of dataset or below minimum (6 samples at least) (asked {val}!)")

    return val



class MedNISTDataset(DatasetManager):
    """"""
    def __init__(self, db, folder_path: str, random_seed: Optional[int] = None):
        """_summary_

        Args:
            folder_path: folder path of the dataset
            random_seed: _description_. Defaults to None.
        """
        super().__init__(db=db)
        self._random_seed: int = random_seed

        self._folder_path: str = folder_path

        if not os.path.exists(folder_path):
            # download ednist dataset
            self.load_mednist_database(path=folder_path)
        self.directories = os.listdir(path=folder_path)
        if random_seed is not None:
            # setting random seed
            random.seed(random_seed)
        self.img_paths_collection: Dict[str, List] = {dir: os.listdir(
            os.path.join(self._folder_path, dir)) for dir in self.directories if  os.path.isdir(os.path.join(self._folder_path, dir))}
        self._old_idx: List[int] = [0 for dir in self.directories]

        for item in self.img_paths_collection.values():
            random.shuffle(item)

    def load_mednist_database(self, path: str, as_dataset: bool = False) -> Tuple[List[int] | Any]:
        # little hack that download MedNIST dataset if it is not located in directory and save in the database
        # the sampled values

        val, download_path = super().load_mednist_database(
            Path(path).parent, as_dataset
        )

        dataset = datasets.ImageFolder(path,
                                       transform=transforms.ToTensor())

        val = self.get_torch_dataset_shape(dataset)
        return val, path

    def sample_image_dataset(self,
                             n_samples: int,
                             n_classes: int,
                             new_sampled_dataset_name: str = "") -> str:
        """Samples and creates a dataset from a image dataset.
        Creates from an existing image datasets a sampled dataset that has almost the same amount of samples per classes

        If the sampled dataset already exists, asks the user to delete dataset before sample it once again
        Args:
            n_samples: number of images to sample
            n_classes: number of classes the dataset holds
            new_sampled_dataset_name: name of the new sampled dataset

        Returns:
            str: the path to the sampled dataset
        """

        n_samples_per_class = n_samples // n_classes
        rest = n_samples % n_classes

        new_image_folder_path = os.path.join(Path(self._folder_path).parent, new_sampled_dataset_name)
        _do_sampling = True
        if os.path.exists(new_image_folder_path):
            print(f"Dataset sampled already exists: {new_image_folder_path}")

            _do_sampling = input('Do you want to delete existing dataset (y/n)').lower() == 'y'

            if _do_sampling:
                # delete existing dataset
                shutil.rmtree(new_image_folder_path)
                print("Deletion completed!")
            else:
                # reload the existing sampled dataset
                print(f"Re-loading dataset {new_image_folder_path}")

        if _do_sampling:
            os.makedirs(new_image_folder_path, exist_ok=True)
            dirs = self.directories.copy()
            for i, directory in enumerate(self.directories):

                label_img_path = os.path.join(self._folder_path, directory)
                if not os.path.isdir(label_img_path):
                    # remove the tarball file copied by mistake (if any)
                    dirs.remove(directory)
                    continue
                # images_path = os.listdir(label_img_path)
                # random.shuffle(images_path)

                _new_dir_label_name = os.path.join(new_image_folder_path, directory)
                os.makedirs(_new_dir_label_name, exist_ok=True)

                images_path = self.img_paths_collection[directory]
                _idx_max = min(n_samples_per_class, len(images_path))
                for image_path in images_path[self._old_idx[i]:_idx_max]:

                    shutil.copy2(
                        os.path.join(label_img_path,
                                     image_path),
                        os.path.join(_new_dir_label_name,  image_path)
                        )

                self._old_idx[i] += _idx_max
            while rest > 0:  # here rest < n_classes

                directory = dirs[rest]
                images_path = os.listdir(os.path.join(self._folder_path, directory))
                if _idx_max == len(images_path) - 1:
                    continue
                image_path = images_path[_idx_max]
                shutil.copy2(
                    os.path.join(self._folder_path,
                                 directory,
                                 image_path),
                    os.path.join(new_image_folder_path, directory, image_path)
                )

                self._old_idx[i] += 1
                rest -= 1

        return new_image_folder_path



if __name__ == '__main__':

    args = parse_args()
    root_folder = os.path.abspath(os.path.expanduser(args.root_folder))
    assert os.path.isdir(root_folder), f'Folder does not exist: {root_folder}'
    data_folder = os.path.join(root_folder, 'data')
    os.makedirs(data_folder, exist_ok=True)

    n_nodes = args.number_nodes

    config_files = []
    errors_coll = []
    for n in range(abs(n_nodes)):
        _name = f"MedNIST_{n+1}_sampled"
        print("Now creating Node: ", f"MedNIST_{n+1}")
        n_sample: Union[str, int] = ask_nb_sample_for_mednist_dataset()

        if n_sample != '':
            config =  manage_config_file(_name)#_name, config_files)
            dataset = MedNISTDataset(
                os.path.join(config.root, 'etc', config.get('default', 'db')),
                os.path.join(data_folder, 'MedNIST'),
                args.random_seed)
            d_path = dataset.sample_image_dataset(
                                                  int(n_sample),
                                                  n_classes=6,
                                                  new_sampled_dataset_name=_name)
        else:
            _name = f"MedNIST_{n+1}"
            config = manage_config_file(_name)
            d_path = os.path.join(data_folder, 'MedNIST')
            dataset = MedNISTDataset(
                os.path.join(config.root, 'etc', config.get('default', 'db')), d_path)

        try:
            dataset.add_database(_name,
                                 'mednist',
                                 ['#MEDNIST', "#dataset"],
                                 description="MedNIST dataset for transfer learning",
                                 path=d_path,
                                 )
        except FedbiomedDatasetManagerError as e:
            errors_coll.append((_name, e))
        config_files.append(config)
    if errors_coll:
        print("Cannot generate dataset because a dataset has been peviously created.")
        for wrong_conf, e in errors_coll:

            print(f"please run:\n fedbiomed node --path=./node_{wrong_conf} dataset delete\nand remove the dataset tagged as {wrong_conf}")
        print("\n\n\n")
        raise e
    print("Done ! please find below your config files:")

    for entry in config_files:
        print("config file: ", os.path.basename(entry.root) + '\n')

    print("to launch the node, please run in distinct terminal:")
    for entry in config_files:
        print(f"fedbiomed node --path {os.path.basename(entry.root)} start")
