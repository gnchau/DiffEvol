import argparse
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.special
from scipy.sparse import csr_matrix
from scipy.signal import find_peaks
from scipy.stats import entropy as _entropy
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

MUTATION_RATE = 1e-5
PSEUDOCOUNT = 1e-8
MIN_BLOCK_LENGTH = 14
WINDOW_MULTIPLIER = 3

CONFIGS = {
    "covid": {
        "label": "SARS-CoV-2",
        "records": "records_8_5M.npz",
        "freq_cache": "covid_filtered.npz",
        "evo_cache": "evo_results_full.npz",
        "pipeline": {"min_coverage": 100, "start_date": 20200101, "end_date": 20230529},
        "warmup": 14,
        "smooth": 7,
        "month_interval": 3,
        "reference_dates": [("2020-12-15", "Vaccine rollout start", "#7b3294"),
                            ("2021-11-15", "Omicron emergence", "#e66101")],
        "out_stem": "figure3_covid_freq_vs_constraint",
    },
    "flu": {
        "label": "Influenza A (H1N1)",
        "fasta": "gisaid_epiflu_sequence.fasta",
        "xls": "gisaid_epiflu_isolates.xls",
        "segment": "HA",
        "freq_cache": "flu_filtered.npz",
        "evo_cache": "flu_evo_results.npz",
        "pipeline": {"min_coverage": 0, "min_daily_coverage": 15, "date_bin_days": 7},
        "warmup": 2,
        "smooth": 2,
        "month_interval": 12,
        "reference_dates": [],
        "out_stem": "figureS_flu_freq_vs_constraint",
    },
}

FIG_WIDTH_IN = 6.5
FIG_HEIGHT_IN = 8.0
DPI = 600
FS_TICK = 8
FS_LABEL = 9
FS_TITLE = 9
FS_LEGEND = 8
COLOR_FREQ = "#8fb8de"
COLOR_K = "#08306b"
COLOR_PEAK = "#d62728"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": FS_TICK,
    "axes.labelsize": FS_LABEL,
    "axes.titlesize": FS_TITLE,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "legend.fontsize": FS_LEGEND,
    "pdf.fonttype": 42,
})


def load_records_numpy(filename):
    data = np.load(filename, allow_pickle=True)
    return [[str(d), str(l), str(s)]
            for d, l, s in zip(data["dates"], data["locations"], data["sequences"])]


def parse_date_to_int(date_str):
    if date_str is None:
        return 0
    s = str(date_str).strip()
    if s == "" or s.lower() in ("unknown", "nan", "nat", "none"):
        return 0
    s = s.split(" ")[0].split("T")[0]
    if re.search(r"(^|-)00(-|$)", s):
        return 0
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y", "%d%b%Y", "%d-%b-%Y"):
        try:
            return int(datetime.strptime(s, fmt).strftime("%Y%m%d"))
        except ValueError:
            continue
    return 0


def parse_records(records):
    sequence_to_id, unique_sequences, processed = {}, [], []
    for date_str, _, sequence in tqdm(records, desc="Processing records"):
        if sequence not in sequence_to_id:
            sequence_to_id[sequence] = len(unique_sequences)
            unique_sequences.append(sequence)
        processed.append([parse_date_to_int(date_str), sequence_to_id[sequence]])
    return unique_sequences, processed


def rebin_dates(dates_int, period_days):
    dates_int = np.asarray(dates_int)
    if period_days <= 1:
        return dates_int
    unique_vals, inverse = np.unique(dates_int, return_inverse=True)
    rebinned = np.empty_like(unique_vals)
    for i, d in enumerate(unique_vals):
        ordinal = datetime.strptime(str(int(d)), "%Y%m%d").toordinal()
        bucket = datetime.fromordinal((ordinal // period_days) * period_days)
        rebinned[i] = int(bucket.strftime("%Y%m%d"))
    return rebinned[inverse]


def build_occurrence_matrix(processed, num_sequences):
    arr = np.array(processed)
    dates, seq_ids = arr[:, 0], arr[:, 1]
    unique_dates = np.sort(np.unique(dates))
    date_to_index = {d: i for i, d in enumerate(unique_dates)}
    rows = [date_to_index[d] for d in dates]
    matrix = csr_matrix((np.ones(len(arr)), (rows, seq_ids)),
                        shape=(len(unique_dates), num_sequences))
    return matrix, unique_dates


def filter_dates_and_coverage(matrix, dates, sequences, min_coverage, start_date, end_date):
    mask = (dates >= start_date) & (dates <= end_date)
    matrix, dates = matrix[mask, :], dates[mask]
    coverage = np.asarray(matrix.sum(axis=0)).reshape(-1)
    keep = coverage >= max(min_coverage, 1)
    print(f"Time points: {len(dates)}; variants: {len(sequences)} -> {int(keep.sum())}")
    return matrix[:, keep], dates, [s for s, k in zip(sequences, keep) if k]


def trim_leading_sparse_period(matrix, dates, min_daily_coverage, smooth_window=14):
    totals = np.asarray(matrix.sum(axis=1)).reshape(-1)
    if min_daily_coverage <= 0 or len(totals) < smooth_window * 2:
        return matrix, dates
    smoothed = np.convolve(totals, np.ones(smooth_window) / smooth_window, mode="same")
    above = np.where(smoothed >= min_daily_coverage)[0]
    if len(above) == 0:
        return matrix, dates
    cut = int(above[0])
    if cut:
        print(f"Trimmed {cut} leading sparse time points")
    return matrix[cut:], dates[cut:]


def trans_mat(m, sequences):
    max_len = max(len(s) for s in sequences)
    padded = [s.ljust(max_len, "N") for s in sequences]
    G, L = len(padded), max_len
    seq_array = np.array([list(s) for s in padded], dtype="U1")
    log_1_m, log_m = np.log(1 - m), np.log(m)
    Mlog = np.zeros((G, G))
    for i in tqdm(range(G), desc="Building transition matrix"):
        h = (seq_array[i] != seq_array).sum(axis=1)
        Mlog[i] = (L - h) * log_1_m + h * log_m
    return scipy.special.softmax(Mlog, axis=1)


def constraint_ker(freq, W, pseudocount=PSEUDOCOUNT):
    T, G = freq.shape
    if pseudocount > 0:
        f0 = (1 - pseudocount * G) * freq[0] + pseudocount
        f0 = f0 / f0.sum()
    else:
        f0 = freq[0].copy()
    K = np.zeros((T, G))
    g_entropy = np.zeros(T)
    g_t = f0.copy()
    for t in tqdm(range(T), desc="Computing constraint matrix"):
        if t > 0:
            g_t = g_t @ W
        g_safe = np.maximum(g_t, 1e-15)
        g_entropy[t] = _entropy(g_safe / g_safe.sum())
        ratio = freq[t] / g_safe
        K[t] = ratio / (ratio.sum() + 1e-15)
    return K, g_entropy


def run_pipeline(records, min_coverage=10, start_date=None, end_date=None,
                 min_daily_coverage=0, date_bin_days=1, mutation_rate=MUTATION_RATE):
    sequences, processed = parse_records(records)
    processed = [r for r in processed if r[0] != 0]
    if date_bin_days > 1:
        rebinned = rebin_dates([r[0] for r in processed], date_bin_days)
        for r, d in zip(processed, rebinned):
            r[0] = int(d)
    matrix, dates = build_occurrence_matrix(processed, len(sequences))
    start_date = int(dates.min()) if start_date is None else start_date
    end_date = int(dates.max()) if end_date is None else end_date
    matrix, dates, sequences = filter_dates_and_coverage(
        matrix, dates, sequences, min_coverage, start_date, end_date)
    matrix, dates = trim_leading_sparse_period(matrix, dates, min_daily_coverage)
    dense = matrix.toarray().astype(float)
    freq = dense / (dense.sum(axis=1, keepdims=True) + 1e-15)
    W = trans_mat(mutation_rate, sequences)
    K, g_entropy = constraint_ker(freq, W)
    return {"freq": freq, "K": K, "dates": dates, "g_entropy": g_entropy, "sequences": sequences}


def load_or_build(cfg):
    if Path(cfg["freq_cache"]).exists() and Path(cfg["evo_cache"]).exists():
        print(f"Loading cached results: {cfg['freq_cache']}, {cfg['evo_cache']}")
        freq = np.load(cfg["freq_cache"], allow_pickle=True)["freq_matrix"]
        evo = np.load(cfg["evo_cache"], allow_pickle=True)
        return freq, evo["K"], evo["dates"]

    if "records" in cfg:
        records = load_records_numpy(cfg["records"])
    else:
        from load_gisaid import load_gisaid
        records = load_gisaid(cfg["fasta"], cfg["xls"], segment_filter=cfg["segment"])
    res = run_pipeline(records, **cfg["pipeline"])
    np.savez(cfg["freq_cache"], freq_matrix=res["freq"])
    np.savez(cfg["evo_cache"], K=res["K"], dates=res["dates"], g_entropy=res["g_entropy"],
             sequences=np.array(res["sequences"], dtype=object))
    print(f"Saved {cfg['freq_cache']} and {cfg['evo_cache']}")
    return res["freq"], res["K"], res["dates"]


def to_dist(x, eps=1e-15):
    x = np.clip(np.asarray(x, dtype=float), 0, None)
    s = x.sum()
    return x / s if s > eps else np.full_like(x, 1.0 / len(x))


def jensen_shannon_divergence(p, q):
    p, q = to_dist(p), to_dist(q)
    m = 0.5 * (p + q)
    return 0.5 * _entropy(p, m) + 0.5 * _entropy(q, m)


def estimate_autocorrelation_time(x, smooth_window=7, c=5):
    x = np.asarray(x, dtype=float)
    if smooth_window > 1 and len(x) > smooth_window * 2:
        x = np.convolve(x, np.ones(smooth_window) / smooth_window, mode="same")
    x = x - x.mean()
    n = len(x)
    max_lag = max(n // 3, 2)
    var = np.dot(x, x) / n
    if var <= 0:
        return 1
    rho = np.array([np.dot(x[:-lag], x[lag:]) / ((n - lag) * var)
                    for lag in range(1, max_lag + 1)])
    tau = 1.0
    for m in range(1, max_lag + 1):
        tau = 1 + 2 * np.sum(rho[:m])
        if m >= c * tau:
            break
    return max(int(round(tau)), 1)


def estimate_block_length(K):
    block = estimate_autocorrelation_time(K.mean(axis=1)) * 2
    T = K.shape[0]
    return int(np.clip(block, MIN_BLOCK_LENGTH, max(MIN_BLOCK_LENGTH, T // 4)))


def compute_D_curve(M, window):
    T = M.shape[0]
    m_tilde = np.array([to_dist(M[t]) for t in range(T)])
    taus = np.arange(window, T - window)
    D = np.array([jensen_shannon_divergence(m_tilde[t - window:t].mean(axis=0),
                                            m_tilde[t:t + window].mean(axis=0))
                  for t in taus])
    return taus, D


def find_notable_peaks(taus, D, prominence_frac=0.5, distance=30, top_n=8):
    if len(D) == 0:
        return np.array([], dtype=int), np.array([])
    idx, props = find_peaks(D, prominence=prominence_frac * np.std(D), distance=distance)
    if len(idx) == 0:
        return np.array([], dtype=int), np.array([])
    keep = np.argsort(-props["prominences"])[:top_n]
    return taus[idx[keep]], D[idx[keep]]


def gini(x, eps=1e-15):
    x = np.clip(x, 0, None)
    if x.sum() <= eps or np.count_nonzero(x) <= 1:
        return np.nan
    sx = np.sort(x)
    n = len(x)
    return (2 * np.sum(np.arange(1, n + 1) * sx)) / (n * sx.sum()) - (n + 1) / n


def smooth_rows(M, window):
    if window <= 1:
        return M
    kernel = np.ones(window) / window
    out = np.column_stack([np.convolve(M[:, g], kernel, mode="same") for g in range(M.shape[1])])
    sums = out.sum(axis=1, keepdims=True)
    return out / np.where(sums > 0, sums, 1.0)


def compute_diagnostics(freq, K, warmup, smooth):
    K_trim, freq_trim = K[warmup:], freq[warmup:]
    K_s, freq_s = smooth_rows(K_trim, smooth), smooth_rows(freq_trim, smooth)
    T = K_s.shape[0]
    window = max(MIN_BLOCK_LENGTH, estimate_block_length(K_trim) * WINDOW_MULTIPLIER)
    tau_K, D_K = compute_D_curve(K_trim, window)
    tau_f, D_f = compute_D_curve(freq_trim, window)
    peak_taus, peak_vals = find_notable_peaks(tau_K, D_K)
    return {
        "max": (freq_s.max(axis=1), K_s.max(axis=1)),
        "entropy": (np.array([_entropy(to_dist(freq_s[t])) for t in range(T)]),
                    np.array([_entropy(to_dist(K_s[t])) for t in range(T)])),
        "gini": (np.array([gini(freq_s[t]) for t in range(T)]),
                 np.array([gini(K_s[t]) for t in range(T)])),
        "D": (tau_f, D_f, tau_K, D_K),
        "peaks": (peak_taus, peak_vals),
    }


def plot_comparison(freq, K, dates_int, cfg):
    dates_all = [datetime.strptime(str(int(d)), "%Y%m%d") for d in dates_int]
    dates = dates_all[cfg["warmup"]:]
    diag = compute_diagnostics(freq, K, cfg["warmup"], cfg["smooth"])
    refs = [(datetime.strptime(d, "%Y-%m-%d"), lbl, col) for d, lbl, col in cfg["reference_dates"]]

    fig, axes = plt.subplots(4, 1, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), sharex=True,
                             constrained_layout=True)
    ax_max, ax_D, ax_H, ax_G = axes

    for ax, key in ((ax_max, "max"), (ax_H, "entropy"), (ax_G, "gini")):
        f_series, k_series = diag[key]
        ax.plot(dates, f_series, color=COLOR_FREQ, linewidth=1.0)
        ax.plot(dates, k_series, color=COLOR_K, linewidth=1.3)

    tau_f, D_f, tau_K, D_K = diag["D"]
    ax_D.plot([dates[t] for t in tau_f], D_f, color=COLOR_FREQ, linewidth=1.0)
    ax_D.plot([dates[t] for t in tau_K], D_K, color=COLOR_K, linewidth=1.3)
    peak_taus, peak_vals = diag["peaks"]
    if len(peak_taus):
        ax_D.scatter([dates[t] for t in peak_taus], peak_vals, color=COLOR_PEAK, s=14, zorder=5)

    titles = ["A  Dominant-genotype concentration", "B  Constraint-shift signal",
              "C  Genotype diversity", "D  Selection concentration"]
    ylabels = ["Max. genotype weight", r"$D(\tau)$ (JSD)", "Entropy (nats)", "Gini coefficient"]
    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        for d, _, col in refs:
            ax.axvline(d, color=col, linestyle="--", linewidth=1.1)
    ax_G.set_xlim(dates[0], dates[-1])
    ax_G.xaxis.set_major_locator(mdates.MonthLocator(interval=cfg["month_interval"]))
    ax_G.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax_G.get_xticklabels(), rotation=45, ha="right")
    ax_G.set_xlabel("Date")

    handles = [Line2D([], [], color=COLOR_FREQ, linewidth=1.5, label="Observed frequency"),
               Line2D([], [], color=COLOR_K, linewidth=1.5, label="DiffEvol constraint"),
               Line2D([], [], color=COLOR_PEAK, marker="o", linestyle="", markersize=4,
                      label="Selected peak")]
    handles += [Line2D([], [], color=col, linestyle="--", linewidth=1.1, label=lbl)
                for _, lbl, col in refs]
    fig.legend(handles=handles, loc="outside upper center", ncol=min(len(handles), 3),
               frameon=False)

    fig.savefig(f"{cfg['out_stem']}.pdf")
    fig.savefig(f"{cfg['out_stem']}.png", dpi=DPI)
    print(f"Saved {cfg['out_stem']}.pdf and {cfg['out_stem']}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--virus", choices=sorted(CONFIGS), default="covid")
    args = parser.parse_args()
    cfg = CONFIGS[args.virus]
    freq, K, dates = load_or_build(cfg)
    plot_comparison(freq, K, dates, cfg)


if __name__ == "__main__":
    main()
