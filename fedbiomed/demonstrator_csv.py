
from fedbiomed.common.constants import TrainingPlans
from fedbiomed.common.data import DataManager
from fedbiomed.common.data._tabular_dataset import CSVDataset
from fedbiomed.common.data.readers import CSVReader




path = 'dataset/CSV/pseudo_adni_mod.csv'



#csv = pd.read_csv(path, delimiter=';')
csv = CSVDataset(path)
dm = DataManager(csv)

#dm.load(tp_type=TrainingPlans.TorchTrainingPlan)
dm.load(tp_type=TrainingPlans.SkLearnTrainingPlan)
val = next(iter(dm.dataset))
print(val)