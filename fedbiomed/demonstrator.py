
from fedbiomed.common.data import MedicalFolderDataset


index_col = 13
path = 'dataset/example/'
demographics_path = 'dataset/example/participants.csv'

demographics_transform = lambda x: x['WEIGHT']
dataset = MedicalFolderDataset(path, ['T1', 'T2'],  transform={'T1': lambda x:x , 'T2': lambda x:x}, target_modalities='label',
                               demographics_transform=demographics_transform,
                               tabular_file=demographics_path, index_col=index_col)

dataset.to_sklearn()
print(dataset._demographics_transform)
print(dataset[0])