from pathlib import Path

import numpy as np
import pandas as pd


def simulate_combat_data(
    n_covariates=5, n_phenotypes=2, n_samples=100, n_batches=3, seed=None
):
    """
    Simulates data following the ComBat model:
        Y_ij = alpha_j + X_i theta_j + gamma_{b(i)j} + delta_{b(i)j} * eps_ij

    Parameters
    ----------
    n_covariates : int
        Number of covariates X.
    n_phenotypes : int
        Number of phenotypes / features Y.
    n_samples : int
        Total number of observations.
    n_batches : int
        Number of batches / sites.
    seed : int or None
        Random seed.

    Returns
    --------
    data : dict with
        X : (n_samples, n_covariates) covariates
        Y : (n_samples, n_phenotypes) data with batch effect
        batch : (n_samples,) batch indices (0..n_batches-1)
        alpha : (n_phenotypes,) global intercept
        theta : (n_covariates, n_phenotypes) biological effects
        gamma : (n_batches, n_phenotypes) additive batch effects
        delta : (n_batches, n_phenotypes) multiplicative batch effects (>0)
        eps : (n_samples, n_phenotypes) standard normal noise
    """
    rng = np.random.default_rng(seed)

    # Split samples across batches
    base = n_samples // n_batches
    sizes = np.full(n_batches, base, dtype=int)
    sizes[: n_samples - base * n_batches] += 1  # distribute the remainder
    batch = np.concatenate([np.full(s, b, dtype=int) for b, s in enumerate(sizes)])

    # Covariates X (Gaussian, centered)
    X = rng.normal(size=(n_samples, n_covariates))

    # "True" parameters
    alpha = rng.normal(loc=0.0, scale=1.0, size=(n_phenotypes,))
    theta = rng.normal(loc=0.0, scale=0.5, size=(n_covariates, n_phenotypes))
    # theta = rng.normal(loc=0.0, scale=0.1,
    #               size=(n_covariates, n_phenotypes))

    # Additive batch effects gamma
    # gamma = rng.normal(loc=0.0, scale=1.0, size=(n_batches, n_phenotypes))
    gamma = rng.normal(loc=0.0, scale=5.0, size=(n_batches, n_phenotypes))

    # Multiplicative batch effects delta (around 1, positive)
    # delta_raw = rng.normal(size=(n_batches, n_phenotypes))
    delta_raw = rng.normal(loc=0.0, scale=2.0, size=(n_batches, n_phenotypes))
    delta = 1.0 + delta_raw
    delta = np.clip(delta, 0.2, 3.0)

    # Noise
    # eps = 0
    # eps = rng.normal(size=(n_samples, n_phenotypes))
    eps = rng.normal(scale=0.3, size=(n_samples, n_phenotypes))

    # Build Y
    Y = np.empty((n_samples, n_phenotypes))
    XB = X @ theta  # (n_samples, n_phenotypes)

    for i in range(n_samples):
        b = batch[i]
        Y[i] = alpha + XB[i] + gamma[b] + delta[b] * eps[i]

    return {
        "X": X,
        "Y": Y,
        "batch": batch,
        "alpha": alpha,
        "theta": theta,
        "gamma": gamma,
        "delta": delta,
        "eps": eps,
    }


# Generate data for FedComBat test

gen_data = simulate_combat_data(
    n_covariates=3, n_phenotypes=2, n_samples=600, n_batches=2, seed=123
)

X = gen_data["X"]
Y = gen_data["Y"]
batch = gen_data["batch"]

# Names
covariate_names = ["SEX", "AGE", "PTEDUCAT"]
phenotype_names = ["CDRSB.bl", "RAVLT.forgetting.bl"]

# Output directory is script's directory
output_dir = Path(__file__).resolve().parent

# Merge X and Y into one table
all_columns = covariate_names + phenotype_names
all_values = np.concatenate([X, Y], axis=1)

# Create one dataframe per batch
batch_dataframes = {}

for b in np.unique(batch):
    # Select rows for this batch
    mask = batch == b

    # Create dataframe
    df_batch = pd.DataFrame(all_values[mask], columns=all_columns)

    # Store in dict
    batch_dataframes[b] = df_batch

    # Save CSV
    csv_path = output_dir / f"batch_{b}.csv"
    df_batch.to_csv(csv_path, index=False)

    print(f"Saved: {csv_path}")
