import os

import numpy as np
from sklearn import datasets

from ._helpers import get_data_folder


def generate_controlled_analytics_dataset():
    """Two-node CSV dataset with analytically-exact statistics.

    200 rows total (100 per node), 2 columns:
        A — integers [1…200]:  node 1 gets [1…100],   node 2 gets [101…200]
        B — integers [2…400]:  node 1 gets [2…200],   node 2 gets [202…400]  (= 2·A)

    Combined ground truth (sample statistics, ddof=1, N=200):
        A: count=200, mean=100.5, variance=3350.0,  std=√3350.0
        B: count=200, mean=201.0, variance=13400.0, std=√13400.0

    Derivation: variance([1…N], ddof=1) = N(N+1)/12.
    For N=200 → 200·201/12 = 3350 exactly; variance_B = 4·variance_A = 13400.

    Returns:
        Tuple (path_node1, path_node2).
    """
    path = get_data_folder("controlled-analytics-e2e")
    p1 = os.path.join(path, "node1.csv")
    p2 = os.path.join(path, "node2.csv")

    if os.path.isfile(p1) and os.path.isfile(p2):
        return p1, p2

    a = np.arange(1, 201, dtype=np.float64)
    for out_path, rows in ((p1, a[:100]), (p2, a[100:])):
        np.savetxt(
            out_path,
            np.column_stack([rows, 2 * rows]),
            delimiter=",",
            header="A,B",
            comments="",
        )

    return p1, p2


def generate_sklearn_classification_dataset():
    """Generate testing data"""

    path = get_data_folder("sklearn-e2e-tests")

    p1 = os.path.join(path, "c1.csv")
    p2 = os.path.join(path, "c2.csv")
    p3 = os.path.join(path, "c3.csv")

    # If there is path stop generating the data again
    if all(os.path.isfile(p) for p in (p1, p2, p3)):
        return p1, p2, p3

    X, y = datasets.make_classification(
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
        random_state=123,
    )
    C1 = X[:150, :]
    C2 = X[150:250, :]
    C3 = X[250:300, :]

    y1 = y[:150].reshape([150, 1])
    y2 = y[150:250].reshape([100, 1])
    y3 = y[250:300].reshape([50, 1])

    n1 = np.concatenate((C1, y1), axis=1)
    np.savetxt(p1, n1, delimiter=",")

    n2 = np.concatenate((C2, y2), axis=1)
    np.savetxt(p2, n2, delimiter=",")

    n3 = np.concatenate((C3, y3), axis=1)
    np.savetxt(p3, n3, delimiter=",")

    return p1, p2, p3
