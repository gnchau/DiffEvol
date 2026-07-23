"""Core DiffEvol computations: mutational transition matrix and constraint kernel."""

from __future__ import annotations

import numpy as np
import scipy.special
from scipy.linalg import eig
from tqdm import tqdm

EPS = 1e-15


def pad_sequences(sequences, pad_char: str = "N"):
    """Right-pad all sequences to a common length so Hamming distances are defined."""
    max_len = max(len(s) for s in sequences)
    min_len = min(len(s) for s in sequences)
    if max_len != min_len:
        print(f"Variable sequence lengths (min={min_len}, max={max_len}); "
              f"padding to {max_len} with '{pad_char}'")
    return [s.ljust(max_len, pad_char) for s in sequences]


def trans_mat(mutation_rate: float, sequences, verbose: bool = True) -> np.ndarray:
    """Build the row-stochastic mutational transition matrix W.

    Entry (i, j) is the softmax over the log-likelihood of mutating sequence i
    into sequence j under an independent per-site mutation rate.
    """
    sequences = pad_sequences(sequences)
    G = len(sequences)
    L = len(sequences[0])

    if verbose:
        print(f"Computing {G}x{G} transition matrix (sequence length L={L})")

    seq_array = np.array([list(s) for s in sequences], dtype="U1")

    log_1_m = np.log(1 - mutation_rate)
    log_m = np.log(mutation_rate)

    M = np.zeros((G, G))
    iterator = tqdm(range(G), desc="Building transition matrix", disable=not verbose)
    for i in iterator:
        h = (seq_array[i] != seq_array).sum(axis=1)
        M[i] = (L - h) * log_1_m + h * log_m

    return scipy.special.softmax(M, axis=1)


def constraint_ker(freq: np.ndarray, W: np.ndarray, T: int,
                   pseudocount: float = 1e-8, verbose: bool = True) -> np.ndarray:
    """Compute the constraint matrix K by inverting unconstrained diffusion.

    The unconstrained expectation at time t is f0 @ W^t, evaluated through the
    eigendecomposition of W rather than repeated matrix powers. K is the
    normalized ratio of observed to expected frequency.

    The uniform pseudocount on f0 gives the unconstrained diffusion support
    everywhere, which keeps the ratio finite for variants that first appear
    after t=0.

    Args:
        freq: (T, G) observed frequencies; each row sums to 1.
        W: (G, G) row-stochastic transition matrix.
        T: number of time points.

    Returns:
        (T, G) constraint matrix.
    """
    G = W.shape[0]

    f0 = (1 - pseudocount * G) * freq[0] + pseudocount
    f0 = f0 / f0.sum()

    if verbose:
        nonzero = np.count_nonzero(freq[0] > 1e-10)
        print(f"f0 has {nonzero}/{G} nonzero variants (pseudocount={pseudocount})")
        print(f"Eigendecomposition of W ({G}x{G})...")

    eigenvalues, P = eig(W)
    P_inv = np.linalg.inv(P)

    if verbose:
        print(f"Max |eigenvalue|: {np.max(np.abs(eigenvalues)):.6f} (expected ~1.0)")

    f0P = f0 @ P

    K = np.zeros((T, G))
    iterator = tqdm(range(T), desc="Computing constraint matrix", disable=not verbose)
    for t in iterator:
        if t == 0:
            g_t = f0.copy()
        else:
            g_t = ((f0P * eigenvalues ** t) @ P_inv).real

        g_t = np.maximum(g_t, EPS)
        ratio = freq[t] / g_t
        K[t] = ratio / (ratio.sum() + EPS)

    return K


def normalize_frequencies(matrix) -> np.ndarray:
    """Doubly-normalize a (T, G) count matrix into a (G, T) frequency matrix.

    Column normalization gives per-timepoint relative frequency; the subsequent
    row normalization standardizes each variant's temporal profile so that rare
    and dominant variants are comparable on one color scale.
    """
    dense = to_dense(matrix)
    freq = dense.T

    freq = freq / (freq.sum(axis=0, keepdims=True) + EPS)
    freq = freq / (freq.sum(axis=1, keepdims=True) + EPS)
    return freq


def column_normalize(matrix) -> np.ndarray:
    """Column-normalize a (T, G) count matrix, returning (G, T)."""
    dense = to_dense(matrix)
    return (dense / (dense.sum(axis=0, keepdims=True) + EPS)).T


def row_normalize(matrix) -> np.ndarray:
    """Row-normalize a (T, G) count matrix so each time point sums to 1."""
    dense = to_dense(matrix)
    return dense / (dense.sum(axis=1, keepdims=True) + EPS)


def calculate_variant_order(freq_matrix: np.ndarray, weighted: bool = True,
                            earliest_last: bool = True) -> np.ndarray:
    """Order variants by mean emergence time.

    Args:
        freq_matrix: (G, T) frequency matrix.
        weighted: weight time indices by frequency rather than using the mean
            index of nonzero entries.
        earliest_last: reverse the order so the earliest variants plot at the
            bottom of a heatmap.
    """
    mean_times = []
    for row in freq_matrix:
        if row.sum() > 0:
            if weighted:
                mean_times.append(np.average(np.arange(len(row)), weights=row))
            else:
                mean_times.append(np.mean(np.where(row > 0)[0]))
        else:
            mean_times.append(np.inf if weighted else 0)

    order = np.argsort(mean_times)
    return order[::-1] if earliest_last else order


def to_dense(matrix) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        return matrix.toarray().astype(float)
    return np.asarray(matrix, dtype=float)
