from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import umap

FREQ_CACHE = "covid_filtered.npz"
EVO_CACHE = "evo_results_full.npz"
OUT_STEM = "figure2_umap_heatmaps"
DPI = 600
UMAP_SEED = 42

FIG_WIDTH_IN = 6.5
FIG_HEIGHT_IN = 7.2
FS_TICK = 11
FS_LABEL = 12
FS_TITLE = 12
FS_PANEL = 14
N_YTICKS = 5
N_XTICKS = 5
BASES = list("ACGT-N")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": FS_TICK,
    "axes.labelsize": FS_LABEL,
    "axes.titlesize": FS_TITLE,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "pdf.fonttype": 42,
})


def load_cached():
    freq = np.load(FREQ_CACHE, allow_pickle=True)["freq_matrix"]
    evo = np.load(EVO_CACHE, allow_pickle=True)
    dates = [datetime.strptime(str(int(d)), "%Y%m%d") for d in evo["dates"]]
    return freq, evo["K"], dates, list(evo["sequences"])


def umap_embedding(sequences):
    base_map = {b: i for i, b in enumerate(BASES)}
    X = np.array([[base_map.get(c, len(BASES)) for c in s] for s in sequences], dtype=np.float32)
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=UMAP_SEED)
    return reducer.fit_transform(X)


def mean_occurrence_year(freq, dates):
    years = np.array([d.year for d in dates], dtype=float)
    return (years[:, None] * freq).sum(axis=0) / (freq.sum(axis=0) + 1e-15)


def panel_letter(ax, letter):
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(-6, 6),
                textcoords="offset points", fontsize=FS_PANEL, fontweight="bold",
                ha="right", va="bottom")


def heatmap(fig, ax, M, vmax, dates, title, cbar_label):
    T, G = M.shape
    im = ax.imshow(M.T, aspect="auto", origin="lower", cmap="viridis",
                   vmin=0, vmax=vmax, interpolation="nearest")
    xt = np.linspace(0, T - 1, N_XTICKS, dtype=int)
    ax.set_xticks(xt)
    ax.set_xticklabels([dates[i].strftime("%Y-%m") for i in xt], rotation=45, ha="right")
    yt = np.linspace(0, G - 1, N_YTICKS, dtype=int)
    ax.set_yticks(yt)
    ax.set_yticklabels([str(i + 1) for i in yt])
    ax.set_xlabel("Date")
    ax.set_ylabel("Variant rank")
    ax.set_title(title, loc="left")
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(cbar_label, fontsize=FS_LABEL)
    cbar.ax.tick_params(labelsize=FS_TICK)


def main():
    freq, K, dates, sequences = load_cached()
    T, G = freq.shape
    print(f"freq {freq.shape}, K {K.shape}")

    print("Running UMAP")
    embedding = umap_embedding(sequences)
    mean_year = mean_occurrence_year(freq, dates)

    mean_emergence = (np.arange(T)[:, None] * freq).sum(axis=0) / (freq.sum(axis=0) + 1e-15)
    order = np.argsort(mean_emergence)
    freq_sorted, K_sorted = freq[:, order], K[:, order]
    vmax = max(np.percentile(freq_sorted, 99.5), np.percentile(K_sorted, 99.5))

    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.06, h_pad=0.08, wspace=0.1, hspace=0.06)
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1.3])
    ax_umap = fig.add_subplot(gs[0, 0])
    ax_freq = fig.add_subplot(gs[0, 1])
    ax_K = fig.add_subplot(gs[1, :])

    sc = ax_umap.scatter(embedding[:, 0], embedding[:, 1], c=mean_year, cmap="plasma",
                         s=3, alpha=0.8, vmin=2020, vmax=2023, linewidths=0, rasterized=True)
    cbar_a = fig.colorbar(sc, ax=ax_umap, pad=0.02, ticks=[2020, 2021, 2022, 2023])
    cbar_a.set_label("Year", fontsize=FS_LABEL)
    cbar_a.ax.tick_params(labelsize=FS_TICK)
    ax_umap.set_xlabel("UMAP1")
    ax_umap.set_ylabel("UMAP2")
    ax_umap.set_xticks([])
    ax_umap.set_yticks([])
    ax_umap.set_title("Genotype UMAP", loc="left")
    panel_letter(ax_umap, "A")

    heatmap(fig, ax_freq, freq_sorted, vmax, dates, "Observed", "Frequency")
    panel_letter(ax_freq, "B")

    heatmap(fig, ax_K, K_sorted, vmax, dates, "DiffEvol constraint", "Constraint")
    panel_letter(ax_K, "C")

    fig.savefig(f"{OUT_STEM}.pdf", dpi=DPI)
    fig.savefig(f"{OUT_STEM}.png", dpi=DPI)
    plt.close(fig)
    print(f"Saved {OUT_STEM}.pdf and {OUT_STEM}.png")


if __name__ == "__main__":
    main()
