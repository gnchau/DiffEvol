#!/usr/bin/env python3
"""Run the DiffEvol pipeline end to end and write Figures 3B, 3C, and 4.

Examples:
    python scripts/run_analysis.py --dataset covid \
        --fasta data/covid.fasta --metadata data/covid.xls

    python scripts/run_analysis.py --dataset flu --records data/flu_records.npz
"""

from __future__ import annotations

import argparse
import sys

from diffevol import (COVID, FLU, generate_fig3bc, generate_fig4, load_gisaid,
                      load_records, prepare_data_for_figures, run_pipeline,
                      save_records)

CONFIGS = {"covid": COVID, "flu": FLU}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=CONFIGS, required=True)

    source = parser.add_argument_group("input (give --records, or both --fasta and --metadata)")
    source.add_argument("--fasta", help="GISAID FASTA file")
    source.add_argument("--metadata", help="GISAID isolates .xls file")
    source.add_argument("--records", help="cached .npz written by a previous run")
    source.add_argument("--save-records", help="cache parsed records to this .npz")
    source.add_argument("--max-records", type=int,
                        help="stop after this many sequences (useful for a smoke test)")

    parser.add_argument("--min-coverage", type=int,
                        help="minimum observations per genotype (default: per dataset)")
    parser.add_argument("--mutation-rate", type=float,
                        help="per-site mutation rate (default: per dataset)")
    parser.add_argument("--date-start", type=int, help="YYYYMMDD")
    parser.add_argument("--date-end", type=int, help="YYYYMMDD")
    parser.add_argument("--vmax-freq", type=float, default=0.03)
    parser.add_argument("--vmax-constraint", type=float,
                        help="default: 99th percentile of nonzero values")
    parser.add_argument("--normalization", choices=("column", "double"), default="column")
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument("--intermediate-dir",
                        help="write genotypes.txt, times.txt, occ_count.npz here")
    parser.add_argument("--skip-fig4", action="store_true")
    parser.add_argument("--n-top", type=int, default=15)

    args = parser.parse_args(argv)

    if not args.records and not (args.fasta and args.metadata):
        parser.error("give --records, or both --fasta and --metadata")

    return args


def main(argv=None):
    args = parse_args(argv)
    config = CONFIGS[args.dataset]

    if args.date_start is not None:
        config.date_start = args.date_start
    if args.date_end is not None:
        config.date_end = args.date_end

    min_coverage = args.min_coverage if args.min_coverage is not None else config.min_coverage
    mutation_rate = args.mutation_rate if args.mutation_rate is not None else config.mutation_rate

    if args.records:
        records = load_records(args.records)
    else:
        records = load_gisaid(args.fasta, args.metadata, max_records=args.max_records)
        if args.save_records:
            save_records(records, args.save_records)

    results = run_pipeline(records, min_coverage=min_coverage,
                           save_dir=args.intermediate_dir)

    prepared = prepare_data_for_figures(results, min_coverage=min_coverage)

    outputs = generate_fig3bc(
        prepared, config,
        mutation_rate=mutation_rate,
        vmax_freq=args.vmax_freq,
        vmax_constraint=args.vmax_constraint,
        normalization=args.normalization,
        output_dir=args.output_dir,
    )

    if not args.skip_fig4:
        generate_fig4(prepared, outputs, config, n_top=args.n_top,
                      output_path=f"{args.output_dir}/{config.prefix}_figure4.png")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
