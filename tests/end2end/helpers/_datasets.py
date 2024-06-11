import os

from sklearn import datasets
import numpy as np

from ._helpers import get_data_folder

def generate_sklearn_classification_dataset():
    """Generate testing data"""

    path = get_data_folder('sklearn-e2e-tests')

    p1 = os.path.join(path , 'c1.csv')
    p2 = os.path.join(path , 'c2.csv')
    p3 = os.path.join(path , 'c3.csv')

    # If there is path stop generating the data again
    if all(os.path.isfile(p) for p in (p1, p2, p3)):
        return p1, p2, p3

    X,y = datasets.make_classification(
        n_samples=300,
        n_features=20,
        n_clusters_per_class=2,
        weights=None,
        flip_y=0.01,
        class_sep=1.0,
        hypercube=True,
        shift=0.0,
        scale=1.0,
        shuffle=True,
        random_state=123
    )
    C1 = X[:150,:]
    C2 = X[150:250,:]
    C3 = X[250:300,:]

    y1 = y[:150].reshape([150,1])
    y2 = y[150:250].reshape([100,1])
    y3 = y[250:300].reshape([50,1])


    n1 = np.concatenate((C1, y1), axis=1)
    np.savetxt(p1, n1, delimiter=',')

    n2 = np.concatenate((C2, y2), axis=1)
    np.savetxt(p2 ,n2, delimiter=',')

    n3 = np.concatenate((C3, y3), axis=1)
    np.savetxt(p3,n3, delimiter=',')


    return p1, p2, p3


