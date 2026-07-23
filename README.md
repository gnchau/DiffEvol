# DiffEvol

Code accompanying **"Constrained Diffusion as a Paradigm for Evolution"**
(Lazarev, Sappington, Chau, Zhang, and Berger; bioRxiv 2026.03.10.710948).

DiffEvol models evolution as diffusion over a discrete genotype space that is
restricted to the set of currently viable genotypes. Given genotype frequencies
over time and the sequences themselves, it inverts the diffusion dynamics to
recover the constraint acting at each time point.

## Details

1. **Transition matrix.** For genotypes `i` and `j` at Hamming distance `h`
   over length `L`, the log-likelihood of mutating `i` into `j` under a
   per-site mutation rate `m` is `(L - h)·log(1 - m) + h·log(m)`. A row-wise
   softmax turns this into a row-stochastic matrix `W`.

2. **Unconstrained diffusion.** Absent any constraint, frequencies would evolve
   as `f₀ Wᵗ`. This is computed from the eigendecomposition of `W` rather than
   by repeated matrix powers, which is both faster and stable since
   `|λ| ≤ 1` for row-stochastic `W`.

3. **Constraint.** The constraint `K[t]` is the observed frequency divided by
   this unconstrained expectation, normalized to sum to one. Genotypes that
   persist above what diffusion alone predicts are the ones under positive
   constraint.

## Installation

```bash
git clone https://github.com/<user>/diffevol.git
cd diffevol
pip install -e .
```

To read GISAID FASTA and XLS files directly, also install the optional:

```bash
pip install -e ".[gisaid]"
```

## Data

**This repository does not include sequence data.** GISAID's terms prohibit
redistribution, so you will need to supply your own.

To reproduce the analyses in the paper, download from
[GISAID](https://gisaid.org/) both:

- a FASTA file of sequences, and
- the corresponding isolates metadata spreadsheet (`.xls`).

Anything reducible to `[date, location, sequence]` triples will work, so the
method is not GISAID-specific — see "Using your own data" below.

## Usage

### Command line

```bash
# From GISAID files, caching parsed records for later runs
python scripts/run_analysis.py --dataset covid \
    --fasta data/covid.fasta \
    --metadata data/covid_isolates.xls \
    --save-records data/covid_records.npz

# From the cache, which skips FASTA parsing entirely
python scripts/run_analysis.py --dataset covid \
    --records data/covid_records.npz \
    --min-coverage 100
```

`--dataset` selects labels, a date window, and defaults; `covid` and `flu` are
predefined. Run with `--help` for the full option list.

Parsing a large FASTA is by far the slowest step, so `--save-records` on the
first run is worth it. Use `--max-records` for a quick smoke test.

### Python

```python
from diffevol import (COVID, generate_fig3bc, generate_fig4,
                      load_gisaid, prepare_data_for_figures, run_pipeline)

records = load_gisaid("data/covid.fasta", "data/covid_isolates.xls")
results = run_pipeline(records, min_coverage=100)

prepared = prepare_data_for_figures(results, min_coverage=100)
outputs = generate_fig3bc(prepared, COVID, mutation_rate=1e-5)
generate_fig4(prepared, outputs, COVID)
```

To use the core computation on its own:

```python
import numpy as np
from diffevol import trans_mat, constraint_ker
from diffevol.core import row_normalize

W = trans_mat(1e-5, sequences)          # (G, G) row-stochastic
freq = row_normalize(counts)            # (T, G), rows sum to 1
K = constraint_ker(freq, W, T=freq.shape[0])
```

### Using your own data

`run_pipeline` takes any iterable of `[date_string, location, sequence]`. Dates
are parsed leniently (`2021-07-01`, `01Jul2021`, `01-Jul-2021`, `2021-07`,
`2021`), and sequences of unequal length are padded with `N`. Define a
`DatasetConfig` for labels and a date window:

```python
from diffevol import DatasetConfig, run_pipeline

MY_VIRUS = DatasetConfig(
    label="My Virus",
    prefix="myvirus",
    date_start=20150101,
    date_end=20231231,
    min_coverage=10,
    mutation_rate=1e-5,
)

results = run_pipeline(my_records, min_coverage=10)
```

### Parameters worth attention

| Parameter | Notes |
|---|---|
| `min_coverage` | Minimum observations per genotype. The single biggest lever on runtime: `W` is `G × G` and the eigendecomposition is `O(G³)`.|
| `mutation_rate` | Per-site rate for `W`. `1e-5` in the paper. |
| `normalization` | `"column"` normalizes per time point; `"double"` also normalizes each genotype's profile across time, making rare genotypes visible at the cost of comparable magnitudes. |
| `vmax_freq` / `vmax_constraint` | Color ceilings. `vmax_constraint=None` uses the 99th percentile of nonzero values. Set both equal to compare panels directly. |
| `pseudocount` | Uniform mass added to `f₀`. Raise it if `K` shows extreme values in early time points. |

## Scaling

The eigendecomposition of `W` dominates runtime and memory. `G` in the low
thousands is comfortable; well beyond that, `W` alone will strain memory at 8
bytes per entry. Control `G` with `min_coverage`.

Note that `run_pipeline` builds one row per *observed* date, not per calendar
day, so gaps in sampling are not represented as gaps in `t`. Since `Wᵗ` treats
`t` as elapsed diffusion time, irregular sampling will distort the inferred
constraint; aggregate to a regular grid beforehand if your sampling is uneven.

## Repository layout

```
diffevol/
├── diffevol/
│   ├── core.py        transition matrix, constraint kernel, normalization
│   ├── pipeline.py    records → occurrence matrix → filtered matrix
│   ├── io.py          GISAID loading, record caching
│   ├── dates.py       date parsing and formatting
│   ├── metrics.py     entropy, Gini, variance, rate of change
│   ├── figures.py     Figure 3B/3C heatmaps, dataset configs
│   ├── figure4.py     Figure 4 multi-panel comparison
│   └── cli.py         command-line entry point
├── scripts/
│   └── run_analysis.py
├── notebooks/
│   └── example.ipynb
└── tests/
```

## Citation

```bibtex
@article{lazarev2026diffevol,
  title   = {Constrained Diffusion as a Paradigm for Evolution},
  author  = {Lazarev, Daniel and Sappington, Anna and Chau, Grant
             and Zhang, Ruochi and Berger, Bonnie},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.03.10.710948}
}
```
