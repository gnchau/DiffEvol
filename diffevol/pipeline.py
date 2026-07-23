"""Turning raw sequence records into a time x genotype occurrence matrix."""

from __future__ import annotations

import numpy as np
import scipy.sparse
from scipy.sparse import csr_matrix
from tqdm import tqdm

from .dates import parse_date_to_int


def parse_records(records):
    """Deduplicate sequences and convert dates to YYYYMMDD integers.

    Returns:
        (unique_sequences, [[date_int, sequence_id], ...], sequence_to_id)
    """
    sequence_to_id = {}
    unique_sequences = []
    processed = []

    for date_str, _location, sequence in tqdm(records, desc="Processing records"):
        sequence_id = sequence_to_id.get(sequence)
        if sequence_id is None:
            sequence_id = len(unique_sequences)
            sequence_to_id[sequence] = sequence_id
            unique_sequences.append(sequence)

        processed.append([parse_date_to_int(date_str), sequence_id])

    return unique_sequences, processed, sequence_to_id


def build_occurrence_matrix(processed_records, num_sequences):
    """Build the sparse (time x genotype) count matrix.

    Rows are consecutive observed dates, not calendar days: dates with no
    observations do not get a row.
    """
    array = np.array(processed_records)
    dates = array[:, 0]
    sequence_ids = array[:, 1]

    unique_dates = np.sort(np.unique(dates))
    date_to_index = {date: idx for idx, date in enumerate(unique_dates)}
    day_indices = [date_to_index[date] for date in dates]

    occurrence_matrix = csr_matrix(
        (np.ones(len(array)), (day_indices, sequence_ids)),
        shape=(len(unique_dates), num_sequences),
    )

    return occurrence_matrix, date_to_index, unique_dates


def filter_by_coverage(occurrence_matrix, sequences, min_coverage=10):
    """Drop genotypes observed fewer than ``min_coverage`` times overall."""
    coverage = np.array(occurrence_matrix.sum(axis=0)).reshape(-1)
    mask = coverage >= min_coverage

    filtered_matrix = occurrence_matrix[:, mask]
    filtered_sequences = [s for s, keep in zip(sequences, mask) if keep]

    print(f"Genotypes with coverage >= {min_coverage}: "
          f"{len(filtered_sequences)}/{len(sequences)}")
    print(f"Filtered matrix shape: {filtered_matrix.shape}")

    return filtered_matrix, filtered_sequences, coverage


def save_outputs(filtered_sequences, unique_dates, filtered_matrix, output_dir="."):
    """Write genotypes, timepoints, and the count matrix to disk."""
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "genotypes.txt", "w") as handle:
        for sequence in filtered_sequences:
            handle.write(sequence + "\n")

    np.savetxt(output_dir / "times.txt", unique_dates)
    scipy.sparse.save_npz(output_dir / "occ_count.npz", filtered_matrix)

    print(f"Wrote genotypes.txt, times.txt, occ_count.npz to {output_dir}")


def run_pipeline(records, min_coverage=10, save_dir=None):
    """Run the full ingestion pipeline.

    Args:
        records: iterable of ``[date_string, location, sequence]``.
        min_coverage: minimum total observations for a genotype to be kept.
        save_dir: if set, write intermediate outputs to this directory.

    Returns:
        dict with the unfiltered occurrence matrix, the coverage-filtered
        matrix, sequences, dates, and per-genotype coverage counts.
    """
    print("Step 1/4: parsing records...")
    unique_sequences, processed, _ = parse_records(records)

    print("\nStep 2/4: building occurrence matrix...")
    occurrence_matrix, _, unique_dates = build_occurrence_matrix(
        processed, len(unique_sequences)
    )

    print("\nStep 3/4: filtering by coverage...")
    filtered_matrix, filtered_sequences, coverage = filter_by_coverage(
        occurrence_matrix, unique_sequences, min_coverage
    )

    if save_dir is not None:
        print("\nStep 4/4: saving intermediate outputs...")
        save_outputs(filtered_sequences, unique_dates, filtered_matrix, save_dir)
    else:
        print("\nStep 4/4: skipped (no save_dir given)")

    return {
        "occurrence_matrix": occurrence_matrix,
        "filtered_matrix": filtered_matrix,
        "unique_sequences": unique_sequences,
        "filtered_sequences": filtered_sequences,
        "unique_dates": unique_dates,
        "coverage": coverage,
    }


def prepare_data_for_figures(results, min_coverage=10,
                             start_date=None, end_date=None):
    """Select a coverage threshold and date window for downstream figures.

    Applying coverage here rather than reusing ``filtered_matrix`` lets one
    pipeline run feed figures at several thresholds.
    """
    occurrence_matrix = results["occurrence_matrix"]
    unique_sequences = results["unique_sequences"]
    unique_dates = np.asarray(results["unique_dates"])
    coverage = results["coverage"]

    if start_date is not None or end_date is not None:
        lo = start_date if start_date is not None else -np.inf
        hi = end_date if end_date is not None else np.inf
        date_mask = (unique_dates >= lo) & (unique_dates <= hi)
        occurrence_matrix = occurrence_matrix[date_mask, :]
        unique_dates = unique_dates[date_mask]
        coverage = np.array(occurrence_matrix.sum(axis=0)).reshape(-1)

    mask = coverage >= min_coverage
    filtered_matrix = occurrence_matrix[:, mask]
    filtered_sequences = [s for s, keep in zip(unique_sequences, mask) if keep]

    print(f"Genotypes passing coverage >= {min_coverage}: {filtered_matrix.shape[1]}")
    print(f"Time points: {filtered_matrix.shape[0]}")
    print(f"Date range: {unique_dates[0]} -> {unique_dates[-1]}")

    return {
        "matrix": filtered_matrix,
        "dates": unique_dates,
        "sequences": filtered_sequences,
    }
