"""Summary statistics comparing observed frequency to DiffEvol constraint."""

from __future__ import annotations

import numpy as np
from scipy.stats import entropy as shannon_entropy

EPS = 1e-15


def mean_plus_std(matrix):
    """Mean plus standard deviation across genotypes at each time point.

    Preferred over the bare mean, which is fixed at 1/G for normalized rows and
    so carries no signal.
    """
    return matrix.mean(axis=1) + matrix.std(axis=1)


def compute_entropy(matrix):
    """Shannon entropy of the genotype distribution at each time point."""
    return np.array([
        shannon_entropy(row / (row.sum() + EPS) + EPS) for row in matrix
    ])


def effective_n(matrix):
    """Inverse Simpson index, the effective number of genotypes."""
    out = np.zeros(matrix.shape[0])
    for t, row in enumerate(matrix):
        p = row / (row.sum() + EPS)
        out[t] = 1.0 / (np.sum(p ** 2) + EPS)
    return out


def rate_of_change(values, smooth_window=14):
    """First difference, box-smoothed."""
    rate = np.diff(values)
    if smooth_window > 1 and len(rate) >= smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        rate = np.convolve(rate, kernel, mode="same")
    return rate


def compute_variance(matrix):
    return np.var(matrix, axis=1)


def compute_gini(matrix):
    """Gini coefficient of the genotype distribution at each time point."""
    T = matrix.shape[0]
    gini = np.zeros(T)
    for t in range(T):
        values = np.sort(matrix[t])
        n = len(values)
        total = values.sum()
        if total <= 0:
            continue
        gini[t] = (2 * np.sum(np.arange(1, n + 1) * values)) / (n * total) - (n + 1) / n
    return gini


def top_n_variants(matrix, n=15):
    """Indices and trajectories of the n genotypes with the highest mean value."""
    order = np.argsort(np.mean(matrix, axis=0))[::-1][:n]
    return order, matrix[:, order]
