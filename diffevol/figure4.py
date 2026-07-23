"""Figure 4: multi-panel comparison of frequency dynamics and DiffEvol constraint.

Panels:
  A. Evolution (mean + std)
  B. Genotype diversity (entropy)
  C. Rate of change
  D. Variance
  E. Selection concentration (Gini, with dGini/dt)
  F. Top-n genotype trajectories, frequency vs. constraint
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from .core import row_normalize, to_dense
from .dates import int_dates_to_datetime, parse_date_int
from .figures import DatasetConfig, apply_style
from .metrics import (compute_entropy, compute_gini, compute_variance,
                      mean_plus_std, rate_of_change, top_n_variants)

FREQ_COLOR = "#4878CF"
CONSTRAINT_COLOR = "#D65F5F"


def _shade_vaccine_window(ax, config: DatasetConfig, alpha=0.20):
    if config.vaccine_window is None:
        return
    start, end = (parse_date_int(d) for d in config.vaccine_window)
    ax.axvspan(start, end, alpha=alpha, color="gray", label="Vaccine Rollout")


def _format_date_axis(ax, interval=4, show_xlabel=True):
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    if show_xlabel:
        ax.set_xlabel("Date", fontsize=9)


def _twin_panel(fig, gs_cell, dt, left_values, right_values, left_label,
                right_label, title, config, zero_line=False):
    """Panel with frequency on the left axis and constraint on the right.

    The two quantities differ by orders of magnitude, so they need separate
    scales to be legible on one set of axes.
    """
    ax = fig.add_subplot(gs_cell)
    ax.plot(dt, left_values, color=FREQ_COLOR, lw=1.0, label="Frequency")
    ax.set_ylabel(left_label, color=FREQ_COLOR, fontsize=9)
    ax.tick_params(axis="y", labelcolor=FREQ_COLOR, labelsize=7)
    if zero_line:
        ax.axhline(0, color="gray", ls="--", lw=0.5)

    ax2 = ax.twinx()
    ax2.plot(dt, right_values, color=CONSTRAINT_COLOR, lw=1.0, label="DiffEvol")
    ax2.set_ylabel(right_label, color=CONSTRAINT_COLOR, fontsize=9)
    ax2.tick_params(axis="y", labelcolor=CONSTRAINT_COLOR, labelsize=7)

    _shade_vaccine_window(ax, config)
    ax.set_title(title, fontsize=10, fontweight="bold")
    _format_date_axis(ax, show_xlabel=False)

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, fontsize=7, loc="upper right")
    return ax


def generate_fig4(prepared, outputs, config: DatasetConfig, n_top=15,
                  start_date=None, end_date=None,
                  output_path=None, dpi=300, smooth_window=14):
    """Generate the six-panel Figure 4.

    Args:
        prepared: dict from ``prepare_data_for_figures``.
        outputs: dict from ``generate_fig3bc``, supplying the constraint matrix.
        config: dataset labels and defaults.
        n_top: number of genotype trajectories in panel F.

    Returns:
        dict of the computed per-timepoint metric series.
    """
    start_date = start_date if start_date is not None else config.date_start
    end_date = end_date if end_date is not None else config.date_end
    output_path = output_path or f"figures/{config.prefix}_figure4.png"

    dates = np.asarray(outputs.get("dates", prepared["dates"]))
    K_full = outputs["constraint_matrix"]
    freq_full = row_normalize(to_dense(prepared["matrix"]))

    # generate_fig3bc may already have dropped time points, so align the
    # frequency matrix to the constraint matrix before masking further.
    if freq_full.shape[0] != K_full.shape[0]:
        all_dates = np.asarray(prepared["dates"])
        keep = np.isin(all_dates, dates)
        freq_full = freq_full[keep]

    mask = (dates >= start_date) & (dates <= end_date)
    dates = dates[mask]
    K = K_full[mask]
    freq = freq_full[mask]
    T, G = K.shape
    dt = int_dates_to_datetime(dates)

    print(f"Figure 4: {T} time points x {G} genotypes "
          f"({dt[0]:%Y-%m-%d} to {dt[-1]:%Y-%m-%d})")

    freq_signal, K_signal = mean_plus_std(freq), mean_plus_std(K)
    freq_ent, K_ent = compute_entropy(freq), compute_entropy(K)
    freq_var, K_var = compute_variance(freq), compute_variance(K)
    freq_gini, K_gini = compute_gini(freq), compute_gini(K)

    freq_rate = rate_of_change(freq_signal, smooth_window)
    K_rate = rate_of_change(K_signal, smooth_window)
    freq_dgini = rate_of_change(freq_gini, smooth_window)
    K_dgini = rate_of_change(K_gini, smooth_window)

    # Derivatives sit between samples, so plot them at interval midpoints.
    dt_mid = [dt[i] + (dt[i + 1] - dt[i]) / 2 for i in range(T - 1)]

    n_top = min(n_top, G)
    _, freq_top = top_n_variants(freq, n_top)
    _, K_top = top_n_variants(K, n_top)

    apply_style()
    fig = plt.figure(figsize=(16, 24))
    gs = fig.add_gridspec(6, 2, hspace=0.55, wspace=0.30,
                          left=0.08, right=0.95, top=0.97, bottom=0.03)

    _twin_panel(fig, gs[0, :], dt, freq_signal, K_signal,
                "Mean + Std (Frequency)", "Mean + Std (DiffEvol)",
                "A. Evolution", config)

    ax = fig.add_subplot(gs[1, :])
    ax.plot(dt, freq_ent, color=FREQ_COLOR, lw=1.0, label="Frequency")
    ax.plot(dt, K_ent, color=CONSTRAINT_COLOR, lw=1.0, label="DiffEvol")
    _shade_vaccine_window(ax, config)
    ax.set_ylabel("Entropy", fontsize=9)
    ax.set_title("B. Genotype Diversity", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
    _format_date_axis(ax, show_xlabel=False)

    _twin_panel(fig, gs[2, :], dt_mid, freq_rate, K_rate,
                "d/dt (Frequency)", "d/dt (DiffEvol)",
                "C. Rate of Change", config, zero_line=True)

    _twin_panel(fig, gs[3, :], dt, freq_var, K_var,
                "Variance (Frequency)", "Variance (DiffEvol)",
                "D. Constraint Variance", config)

    ax = fig.add_subplot(gs[4, :])
    ax.plot(dt, freq_gini, color=FREQ_COLOR, lw=1.0, label="Frequency")
    ax.plot(dt, K_gini, color=CONSTRAINT_COLOR, lw=1.0, label="DiffEvol")
    _shade_vaccine_window(ax, config)
    ax.set_ylabel("Gini Coefficient", fontsize=9)
    ax.tick_params(axis="y", labelsize=7)

    ax2 = ax.twinx()
    ax2.plot(dt_mid, freq_dgini, color=FREQ_COLOR, lw=0.5, alpha=0.4, ls="--")
    ax2.plot(dt_mid, K_dgini, color=CONSTRAINT_COLOR, lw=0.5, alpha=0.4, ls="--")
    ax2.set_ylabel("dGini/dt", fontsize=8, alpha=0.6)
    ax2.tick_params(axis="y", labelsize=6, colors="gray")
    ax2.axhline(0, color="gray", ls=":", lw=0.5, alpha=0.4)

    ax.set_title("E. Selection Concentration", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="lower right")
    _format_date_axis(ax, show_xlabel=True)

    colors = plt.cm.tab20(np.linspace(0, 1, n_top))
    for cell, data, ylabel, subtitle in (
        (gs[5, 0], freq_top, "Frequency", "F(i). Top {n} — Frequency"),
        (gs[5, 1], K_top, "Constraint (DiffEvol)", "F(ii). Top {n} — DiffEvol"),
    ):
        ax = fig.add_subplot(cell)
        for i in range(n_top):
            ax.plot(dt, data[:, i], lw=0.9, alpha=0.85, color=colors[i])
        _shade_vaccine_window(ax, config)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(subtitle.format(n=n_top), fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=7)
        _format_date_axis(ax, show_xlabel=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"Saved {output_path}")
    plt.close()

    return {
        "freq_signal": freq_signal, "K_signal": K_signal,
        "freq_entropy": freq_ent, "K_entropy": K_ent,
        "freq_gini": freq_gini, "K_gini": K_gini,
        "freq_var": freq_var, "K_var": K_var,
    }
