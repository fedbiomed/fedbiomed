from monai.losses.dice import DiceLoss
from monai.networks.nets import UNet
from monai.transforms import (
    AsDiscrete,
    Compose,
    EnsureChannelFirst,
    NormalizeIntensity,
    Resize,
)
from torch.optim import AdamW

from fedbiomed.common.datamanager import DataManager
from fedbiomed.common.dataset import MedicalFolderDataset
from fedbiomed.common.training_plans import TorchTrainingPlan


class UNetTrainingPlan(TorchTrainingPlan):
    def init_model(self, model_args):
        model = UNet(
            spatial_dims=model_args["spatial_dims"],
            in_channels=model_args["in_channels"],
            out_channels=model_args["out_channels"],
            channels=model_args["channels"],
            strides=model_args["strides"],
            num_res_units=model_args["num_res_units"],
            norm=model_args["norm"],
        )
        self.loss_fn = DiceLoss(include_background=False, sigmoid=True)
        return model

    def init_optimizer(self):
        optimizer = AdamW(self.model().parameters())
        return optimizer

    def init_dependencies(self):
        # Here we define the custom dependencies that will be needed by our custom Dataloader
        deps = [
            "from monai.transforms import (Compose, NormalizeIntensity, EnsureChannelFirst, Resize, AsDiscrete)",
            "import torch.nn as nn",
            "import torch.nn.functional as F",
            "from fedbiomed.common.dataset import MedicalFolderDataset",
            "import numpy as np",
            "from torch.optim import AdamW",
            "from monai.networks.nets import UNet",
            "from monai.losses.dice import DiceLoss",
        ]
        return deps

    def training_data(self):
        # The training_data creates the Dataloader to be used for training in the general class Torchnn of fedbiomed
        common_shape = (48, 64, 48)
        training_transform = Compose(
            [
                EnsureChannelFirst(channel_dim="no_channel"),
                Resize(common_shape),
                NormalizeIntensity(),
            ]
        )
        target_transform = Compose(
            [
                EnsureChannelFirst(channel_dim="no_channel"),
                Resize(common_shape),
                AsDiscrete(to_onehot=2),
            ]
        )
        dataset = MedicalFolderDataset(
            data_modalities="T1",
            target_modalities="label",
            transform={"T1": training_transform},
            target_transform={"label": target_transform},
        )
        return DataManager(dataset)

    def training_step(self, data, target):
        # this function must return the loss to backward it
        img = data["T1"]
        output = self.model().forward(img)
        loss = self.loss_fn(output, target["label"])
        return loss

    def testing_step(self, data, target):
        img = data["T1"]
        target = target["label"]
        prediction = self.model().forward(img)
        loss = self.loss_fn(prediction, target)
        return loss
