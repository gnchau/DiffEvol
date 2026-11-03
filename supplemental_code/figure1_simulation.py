import itertools

import numpy as np
import scipy.special
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

L = 5
BASES = ["A", "C", "G", "T"]
N_TIME = 102
NOISE_N = 3000
MUTATION_RATE = 0.01
RNG_SEED = 42
OUT_STEM = "figure1_simulation"
DPI = 600

FIG_WIDTH_IN = 6.5
FIG_HEIGHT_IN = 3.0
FS_TICK = 11
FS_LABEL = 12
FS_TITLE = 12
FS_PANEL = 14

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": FS_TICK,
    "axes.labelsize": FS_LABEL,
    "axes.titlesize": FS_TITLE,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "pdf.fonttype": 42,
})


def build_genotypes():
    genotypes = ["".join(g) for g in itertools.product(BASES, repeat=L)]
    scores = np.array([sum(BASES.index(b) * 4 ** (L - 1 - i) for i, b in enumerate(g))
                       for g in genotypes])
    order = np.argsort(scores)
    genotypes = [genotypes[i] for i in order]
    seq_array = np.array([list(s) for s in genotypes], dtype="U1")
    return genotypes, seq_array


def transition_matrix(seq_array, m):
    G = seq_array.shape[0]
    Mlog = np.zeros((G, G))
    for i in range(G):
        h = (seq_array[i] != seq_array).sum(axis=1)
        Mlog[i] = (L - h) * np.log(1 - m) + h * np.log(m)
    return scipy.special.softmax(Mlog, axis=1)


def true_constraint(G):
    sigma = G / 10.0
    peak_end = int(G * 0.85)
    K = np.zeros((N_TIME, G))
    for t in range(N_TIME):
        center = peak_end * t / (N_TIME - 1)
        K[t] = np.exp(-0.5 * ((np.arange(G) - center) / sigma) ** 2)
        K[t] /= K[t].sum()
    return K


def forward_simulate(K_true, W):
    G = W.shape[0]
    freq = np.zeros((N_TIME, G))
    freq[0] = K_true[0]
    g_t = K_true[0].copy()
    for t in range(1, N_TIME):
        g_t = g_t @ W
        unnorm = K_true[t] * np.maximum(g_t, 1e-15)
        freq[t] = unnorm / unnorm.sum()
    return freq


def add_sampling_noise(freq, rng):
    F_n = np.zeros_like(freq)
    for t in range(N_TIME):
        counts = rng.multinomial(NOISE_N, np.maximum(freq[t], 0)).astype(float)
        s = counts.sum()
        F_n[t] = counts / s if s > 0 else freq[t]
    return F_n


def constraint_ker_sim(freq, W, pseudocount=None):
    G, T = W.shape[0], freq.shape[0]
    if pseudocount is None:
        pseudocount = 1.0 / G
    f0 = (1 - pseudocount * G) * freq[0] + pseudocount
    f0 /= f0.sum()
    K_est = np.zeros((T, G))
    g_t = f0.copy()
    for t in range(T):
        if t > 0:
            g_t = g_t @ W
        ratio = freq[t] / np.maximum(g_t, 1e-15)
        K_est[t] = ratio / (ratio.sum() + 1e-15)
    return K_est


def plot_figure(K_true, F_n, K_est, G):
    K_true, F_n, K_est = (M * 1e3 for M in (K_true, F_n, K_est))
    vmax = max(np.percentile(M, 99.5) for M in (K_true, F_n, K_est))
    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
                             sharey=True, layout="constrained")
    panels = [(K_true, "A", "Ground truth"), (F_n, "B", "Noisy data"), (K_est, "C", "DiffEvol")]
    for ax, (M, letter, title) in zip(axes, panels):
        im = ax.imshow(M.T, aspect="auto", origin="lower", cmap="Blues",
                       vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_xticks([0, 50, 100])
        ax.set_yticks([0, 512, G - 1])
        ax.set_xlabel("Time")
        ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(0, 5),
                    textcoords="offset points", fontsize=FS_PANEL, fontweight="bold",
                    ha="left", va="bottom")
        ax.annotate(title, xy=(0, 1), xycoords="axes fraction", xytext=(15, 5),
                    textcoords="offset points", fontsize=FS_TITLE, ha="left", va="bottom")
    axes[0].set_ylabel("Genotype")
    cbar = fig.colorbar(im, ax=axes, aspect=25, pad=0.015)
    cbar.set_label(r"Weight ($\times 10^{-3}$)", fontsize=FS_LABEL)
    cbar.ax.tick_params(labelsize=FS_TICK)
    fig.savefig(f"{OUT_STEM}.pdf")
    fig.savefig(f"{OUT_STEM}.png", dpi=DPI)
    plt.close(fig)
    print(f"Saved {OUT_STEM}.pdf and {OUT_STEM}.png")


def main():
    rng = np.random.default_rng(RNG_SEED)
    _, seq_array = build_genotypes()
    G = seq_array.shape[0]
    print(f"Building {G}x{G} transition matrix (m={MUTATION_RATE:.0e})")
    W = transition_matrix(seq_array, MUTATION_RATE)
    K_true = true_constraint(G)
    F_n = add_sampling_noise(forward_simulate(K_true, W), rng)
    print("Estimating constraint matrix")
    K_est = constraint_ker_sim(F_n, W)
    plot_figure(K_true, F_n, K_est, G)


if __name__ == "__main__":
    main()
