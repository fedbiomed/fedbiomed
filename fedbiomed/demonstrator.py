
from fedbiomed.common.constants import TrainingPlans
from fedbiomed.common.data import MedicalFolderDataset
from fedbiomed.common.data import DataManager


index_col = 13
path = 'dataset/example/'
demographics_path = 'dataset/example/participants.csv'

demographics_transform = lambda x: x['WEIGHT']
dataset = MedicalFolderDataset(path, ['T1', 'T2'],  transform={'T1': lambda x:x , 'T2': lambda x:x}, target_modalities='label',
                               demographics_transform=demographics_transform,
                               tabular_file=demographics_path, index_col=index_col)

dataset.to_sklearn()  # change here the framework
print(dataset._demographics_transform)
print(dataset[1])


dataset = MedicalFolderDataset(path, ['T1', 'T2'],  transform={'T1': lambda x:x , 'T2': lambda x:x}, target_modalities='label',
                               demographics_transform=demographics_transform,
                               tabular_file=demographics_path, index_col=index_col)
data_manager = DataManager(dataset)
#data_manager.load(tp_type=TrainingPlans.TorchTrainingPlan)
data_manager.load(tp_type=TrainingPlans.SkLearnTrainingPlan)
train, test = data_manager.split(test_ratio=.0, test_batch_size=None)
val = next(iter(train.dataset))
print(val)
print(len(val))
