'''Fedbiomed Researcher to train a federated scikit learn model.
Purpose of the exercise :

Two datasets n1.csv and n2.csv has been generated randomly using a linear transformation A = [ 5 8 9 5 0 ]. We will fit a Stochastic Gradient Regressor to approximate this transformation using Federated Learning.


Extending this notebook to any incremental learning scikit model:

The same federated learning scheme below applies to any sklearn model supporting the method partial_fit():

A family of models could be naturally imported in Fed-BioMed, following the same approach. For example:

    Naive Bayes.
    Logistic regression,
    SVM/SVC (linear and non-linear),
    perceptron,
    KMeans,
    incremental PCA,
    mini batch dictionary learning,
    latent Dirichlet annotation,


Start the network and setting the client up

Before running this notebook:

    You should start the network from fedbiomed-network, as detailed in : https://gitlab.inria.fr/fedbiomed/fedbiomed
    Download n1.csv, n2.csv and n3.csv to some place in your computer from https://gitlab.inria.fr/fedbiomed/fedbiomed/-/tree/develop/notebooks/data
    You need to configure at least 2 nodes:

    Node 1 : ./scripts/fedbiomed_run node add
        Select option 1 to add a csv file to the client
        Choose the name, tags and description of the dataset (you can write 'sk' always and it will be good)
        Pick the .csv file n1.csv .
        Check that your data has been added in node 1 by executing ./scripts/fedbiomed_run node list
        Run the node using ./scripts/fedbiomed_run node start.

    Node 2 : Open a second terminal and run ./scripts/fedbiomed_run node add config n2.ini
        Select option 1 to add a csv file to the client
        Choose the name, tags and description of the dataset (you can write 'sk' always and it will be good)
        Pick the .csv file n2.csv .
        Check that your data has been added in node 2 by executing ./scripts/fedbiomed_run node list config n2.ini
        Run the node using ./scripts/fedbiomed_run node start config n2.ini.

    Node 3 : Open a second terminal and run ./scripts/fedbiomed_run node add config n3.ini
        Select option 1 to add a csv file to the client
        Choose the name, tags and description of the dataset (you can write 'sk' always and it will be good)
        Pick the .csv file n3.csv .
        Check that your data has been added in node 2 by executing ./scripts/fedbiomed_run node list config n3.ini
        Run the node using ./scripts/fedbiomed_run node start config n3.ini.

    Wait until you get Connected with result code 0. it means you are online.'''
import numpy as np
from fedbiomed.common.fedbiosklearn import SGDSkLearnModel
from sklearn.linear_model import SGDRegressor


class SGDRegressorTrainingPlan(SGDSkLearnModel):
    def __init__(self, kwargs):
        super(SGDRegressorTrainingPlan, self).__init__(kwargs)
        self.add_dependency(["from sklearn.linear_model import SGDRegressor"])

    def training_data(self, batch_size=None):
        NUMBER_COLS = 5
        dataset = pd.read_csv(self.dataset_path, header=None, delimiter=',')
        if batch_size == None:
            X = dataset.iloc[:, 0:NUMBER_COLS].values
            y = dataset.iloc[:, NUMBER_COLS]
        else:
            X = dataset.iloc[0:batch_size, 0:NUMBER_COLS].values
            y = dataset.iloc[0:batch_size, NUMBER_COLS]
        return (X, y.values)

model_args = { 'max_iter':1000, 'tol': 1e-3 , 'number_columns': 5 , 'model': 'SGDRegressor' , 'n_features': 5}

training_args = {
    'batch_size': None,
    'lr': 1e-3,
    'epochs': 5,
    'dry_run': False,
    'batch_maxnum': 0
}


from fedbiomed.researcher.experiment import Experiment
from fedbiomed.researcher.aggregators.fedavg import FedAverage

tags =  ['sk']
rounds = 5

exp = Experiment(tags=tags,
                 #clients=None,
                 #model_path=model_file,
                 model_args=model_args,
                 model_class=SGDRegressorTrainingPlan,
                 training_args=training_args,
                 rounds=rounds,
                 aggregator=FedAverage(),
                 client_selection_strategy=None)

exp.run()

# Lets build now a dataset test, A is the linear transformation that has been used to build the csv file datasets.

n_features = 5
testing_samples = 40
rng = np.random.RandomState(1)
A = np.array([[5],
       [8],
       [9],
       [5],
       [0]])

def test_data():
    X_test = rng.randn(testing_samples, n_features).reshape([testing_samples, n_features])
    y_test = X_test.dot(A) + rng.randn(testing_samples).reshape([testing_samples,1])
    return X_test, y_test


X_test, y_test = test_data()

testing_error = []

for i in range(rounds):
    fed_model = SGDRegressor(max_iter=1000, tol=1e-3)
    fed_model.coef_ = exp._aggregated_params[i]['params']['coef_'].copy()
    fed_model.intercept_ = exp._aggregated_params[i]['params']['intercept_'].copy()
    mse = np.mean((fed_model.predict(X_test).ravel() - y_test.ravel())**2)
    print('MSE ', mse)
    testing_error.append(mse)

