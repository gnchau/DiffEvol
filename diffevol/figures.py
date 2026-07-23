"""Figure generation: frequency and constraint heatmaps (Figure 3B/3C)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .core import (calculate_variant_order, column_normalize, constraint_ker,
                   normalize_frequencies, row_normalize, to_dense, trans_mat)
from .dates import format_date_int, is_valid_date_in_range


@dataclass
class FontSizes:
    labels: int = 18
    ticks: int = 14
    title: int = 20
    colorbar: int = 16


@dataclass
class DatasetConfig:
    """Per-dataset labelling and defaults.

    ``label`` appears in figure titles; ``prefix`` in output filenames.
    """
    label: str
    prefix: str
    date_start: int
    date_end: int
    min_coverage: int = 10
    mutation_rate: float = 1e-5
    vaccine_window: tuple | None = None


COVID = DatasetConfig(
    label="SARS-CoV-2",
    prefix="covid",
    date_start=20200101,
    date_end=20230529,
    min_coverage=100,
    vaccine_window=(20201215, 20210315),
)

FLU = DatasetConfig(
    label="Influenza A H1N1",
    prefix="flu",
    date_start=20090101,
    date_end=20231231,
    min_coverage=10,
)


def apply_style():
    """Editable text in vector output, which journals generally require."""
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["svg.fonttype"] = "none"


def plot_heatmap(data, date_labels, title, cbar_label, output_path,
                 vmin=0, vmax=None, cmap="viridis", figsize=(16, 10), dpi=300,
                 max_xticks=10, max_yticks=20, ylabel="Variants",
                 fonts: FontSizes | None = None):
    """Render a (genotypes x time) matrix as a heatmap and save it."""
    fonts = fonts or FontSizes()
    apply_style()

    n_variants, n_dates = data.shape
    frame = pd.DataFrame(data, columns=date_labels, index=np.arange(n_variants))

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(frame, vmin=vmin, vmax=vmax, cmap=cmap,
                cbar_kws={"label": cbar_label}, ax=ax,
                xticklabels=False, yticklabels=False)

    # Step through the labels rather than using every one, and keep the last
    # tick so the axis always terminates at the true end date.
    step = max(1, n_dates // max_xticks)
    xticks = list(range(0, n_dates, step))
    if xticks[-1] != n_dates - 1:
        xticks.append(n_dates - 1)

    ax.set_xticks(xticks)
    ax.set_xticklabels([date_labels[i] for i in xticks],
                       rotation=45, ha="right", fontsize=fonts.ticks)

    yticks = np.linspace(0, n_variants - 1, min(max_yticks, n_variants), dtype=int)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"Variant {i + 1}" for i in yticks], fontsize=fonts.ticks)

    ax.set_xlabel("Date", fontsize=fonts.labels, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=fonts.labels, fontweight="bold")
    ax.set_title(title, fontsize=fonts.title, fontweight="bold", pad=20)

    cbar = ax.collections[0].colorbar
    cbar.set_label(cbar_label, fontsize=fonts.colorbar, fontweight="bold")
    cbar.ax.tick_params(labelsize=fonts.ticks)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"Saved {output_path}")
    plt.close()


def generate_fig3bc(prepared, config: DatasetConfig, mutation_rate=None,
                    vmax_freq=0.03, vmax_constraint=None,
                    normalization="column", pseudocount=1e-8,
                    output_dir="figures", output_prefix="figure3",
                    dpi=300, figsize=(16, 10), cmap="viridis",
                    max_xticks=10, max_yticks=20,
                    fonts: FontSizes | None = None):
    """Generate the frequency heatmap (3B) and constraint heatmap (3C).

    Both panels share one variant ordering so rows correspond between them.

    Args:
        prepared: dict from :func:`prepare_data_for_figures`.
        config: dataset labels and date window.
        normalization: ``"column"`` normalizes each time point; ``"double"``
            additionally normalizes each variant's profile across time.
        vmax_constraint: color ceiling for 3C. ``None`` uses the 99th
            percentile of nonzero values.

    Returns:
        dict with the frequency matrix, constraint matrix, sort order,
        transition matrix, and the surviving dates.
    """
    mutation_rate = mutation_rate if mutation_rate is not None else config.mutation_rate
    fonts = fonts or FontSizes()

    matrix = prepared["matrix"]
    dates_raw = np.asarray(prepared["dates"])
    sequences = prepared["sequences"]

    valid = np.array([is_valid_date_in_range(d, config.date_start, config.date_end)
                      for d in dates_raw])
    if not valid.any():
        raise ValueError(
            f"No valid dates in [{config.date_start}, {config.date_end}]"
        )

    matrix = matrix[valid]
    dates = dates_raw[valid]
    date_labels = [format_date_int(d) for d in dates]

    dense = to_dense(matrix)
    T, G = dense.shape
    print(f"After date filter: {T} time points x {G} genotypes "
          f"({dates[0]} -> {dates[-1]})")

    if normalization == "double":
        freq_gt = normalize_frequencies(dense)
    elif normalization == "column":
        freq_gt = column_normalize(dense)
    else:
        raise ValueError(f"Unknown normalization: {normalization}")

    sort_order = calculate_variant_order(freq_gt)

    print("\n--- Figure 3B: frequency ---")
    plot_heatmap(
        freq_gt[sort_order], date_labels,
        title=f"Frequency Data — {config.label}",
        cbar_label="Normalized Frequency",
        vmax=vmax_freq,
        output_path=f"{output_dir}/{config.prefix}_{output_prefix}B.png",
        figsize=figsize, dpi=dpi, cmap=cmap,
        max_xticks=max_xticks, max_yticks=max_yticks, fonts=fonts,
    )

    print("\n--- Figure 3C: constraint ---")
    W = trans_mat(mutation_rate, sequences)
    K = constraint_ker(row_normalize(dense), W, T, pseudocount=pseudocount)
    print(f"K range: [{K.min():.4e}, {K.max():.4e}]")

    sorted_K = K.T[sort_order]

    if vmax_constraint is None:
        nonzero = sorted_K[sorted_K > 1e-10]
        vmax_constraint = float(np.percentile(nonzero, 99)) if nonzero.size else 1.0
        print(f"Auto vmax_constraint (99th pctl): {vmax_constraint:.6f}")

    plot_heatmap(
        sorted_K, date_labels,
        title=f"DiffEvol Constraint Matrix — {config.label}",
        cbar_label="Constraint Value",
        vmax=vmax_constraint,
        output_path=f"{output_dir}/{config.prefix}_{output_prefix}C.png",
        figsize=figsize, dpi=dpi, cmap=cmap,
        max_xticks=max_xticks, max_yticks=max_yticks, fonts=fonts,
    )

    return {
        "freq_matrix": freq_gt,
        "constraint_matrix": K,
        "sort_order": sort_order,
        "W": W,
        "dates": dates,
    }
