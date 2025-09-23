from fedbiomed.common.dataset_controller._controller import Controller

"""

- Override the len in CustomDataset (An abstract len method)
- 


"""


class CustomController(Controller):
    """Custom dataset controller for MNIST dataset"""

    def shape(self):
        """Len and shape should never be used, as the dataset implements it specifically.

        It is impossible for the custom controller to know the shape of the dataset.
        """
        return None

    def get_sample(self, index):
        """Shouldn't be called as the custom controller does not need to get a sample.

        It is handled by get_item in CustomDataset.
        """
        return None

    def get_types(self):
        """Controller does not know how to get samples."""
        pass

    def __len__(self) -> int:
        """Similar to shape, should never be used."""
        pass
