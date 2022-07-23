from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.constants import TrainingPlans
from fedbiomed.common.data import DataManager
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedRoundError
from fedbiomed.common.logger import logger


def _set_training_testing_data_loaders_flamby(dataset, model, testing_arguments, transform_compose_flamby, batch_size):
    """Method for setting FLamby training and validation data loaders based on the training and validation
    arguments.

    Args:
        dataset: dataset details to use in this round (dataset name, dataset's id, dataset parameters...)
        model: TrainingPlan model
        testing_arguments: Validation arguments
        transform_compose_flamby: Compose function to optionally perform transformations on the dataset
        batch_size: Number of samples processed before the FLamby model is updated

    Returns:
        TrainingPlan model with configured validation and training parts
    """

    # Get validation parameters
    test_ratio = testing_arguments.get('test_ratio', 0)
    test_global_updates = testing_arguments.get('test_on_global_updates', False)
    test_local_updates = testing_arguments.get('test_on_local_updates', False)

    # Inform user about mismatch arguments settings
    if test_ratio != 0 and test_local_updates is False and test_global_updates is False:
        logger.warning("Validation will not be perform for the round, since there is no validation activated. "
                        "Please set `test_on_global_updates`, `test_on_local_updates`, or both in the "
                        "experiment.")

    if test_ratio == 0 and (test_local_updates is False or test_global_updates is False):
        logger.warning('There is no validation activated for the round. Please set flag for `test_on_global_updates`'
                        ', `test_on_loc_set_training_testing_data_loadersal_updates`, or both. Splitting dataset for validation will be ignored')
    
    training_data_loader, testing_data_loader = _split_train_and_test_data_flamby(dataset=dataset, test_ratio=test_ratio, transform_compose_flamby=transform_compose_flamby, batch_size=batch_size)

    # Set models validation and training parts for model
    model.set_data_loaders(train_data_loader=training_data_loader,
                            test_data_loader=testing_data_loader)
    return model

# /!\ The naming of some methods seems still to be ambiguous
# I keep it as it is, but test data refers here to validation data
def _split_train_and_test_data_flamby(dataset: dict, test_ratio: float = 0, transform_compose_flamby=None, batch_size: int=2):
    """Method for splitting training and validation data for a flamby dataset.

    Returns:
        Tuple containing the DataLoaders for the train and the validation federated flamby dataset.
        Validation should be used to perform hyperparameter tuning. That's why the validation set is a subset of the FLamby training set.
        FLamby test set is accessible by setting train=False in the federated class. This test set should be used to perform the final evaluation
        of the model, and the performance reached will be the one retained in the benchmark.
    """
    training_plan_type = TrainingPlans.TorchTrainingPlan # FLamby dataloaders are all and always based on PyTorch
    module = __import__(dataset['dataset_parameters']['fed_class'], fromlist='dummy')
    center_id = dataset['dataset_parameters']['center_id']

    try:
        fed_class_train = module.FedClass(transform=transform_compose_flamby, center=center_id, train=True, pooled=False) # FLamby pytorch dataloader
    except Exception: # Some flamby datasets don't have a transform parameter, so we need to ignore it in this case
        fed_class_train = module.FedClass(center=center_id, train=True, pooled=False)

    train_kwargs = {'batch_size': batch_size, 'shuffle': True}
    data_manager = DataManager(fed_class_train, **train_kwargs)

    # Specific datamanager based on training plan
    try:
        data_manager.load(tp_type=training_plan_type)
    except FedbiomedError as e:
        raise FedbiomedRoundError(f"{ErrorNumbers.FB314.value}: Error while loading data manager; {str(e)}")

    # Split dataset as train and validation
    return data_manager.split(test_ratio=test_ratio)