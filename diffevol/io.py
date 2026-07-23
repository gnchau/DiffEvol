"""Loading sequence records and caching them between runs.

A "record" is a ``[date_string, location, sequence]`` triple. This is the input
format expected by :func:`diffevol.pipeline.run_pipeline`.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def load_gisaid_metadata(xls_path, engine: str = "xlrd") -> dict:
    """Index a GISAID isolates spreadsheet by accession ID.

    Records are indexed under both the EPI_ISL isolate ID and the bare EPI
    segment accession, since FASTA headers may carry either.
    """
    print(f"Loading metadata from {xls_path}...")
    df = pd.read_excel(xls_path, engine=engine)
    df.columns = [c.strip() for c in df.columns]

    def find_col(candidates):
        for candidate in candidates:
            matches = [c for c in df.columns if candidate.lower() in c.lower()]
            if matches:
                return matches[0]
        return None

    date_col = find_col(["Collection_Date", "Collection Date",
                         "Isolate_Collection_date"])
    location_col = find_col(["Location", "Country", "Geo_Location"])
    epi_isl_col = find_col(["EPI_ISL", "Isolate_Id", "Accession_ID", "Accession ID"])
    epi_id_col = find_col(["Segment_Id", "EPI2", "EPI Sequence Accession"])

    print(f"  date={date_col}  location={location_col}  "
          f"isolate={epi_isl_col}  segment={epi_id_col}")

    metadata = {}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Indexing metadata"):
        entry = {
            "date": str(row[date_col]).strip() if date_col else "Unknown",
            "location": str(row[location_col]).strip() if location_col else "Unknown",
        }
        for col in (epi_isl_col, epi_id_col):
            if col and pd.notna(row[col]):
                metadata[str(row[col]).strip()] = entry

    print(f"Metadata index built: {len(metadata):,} entries")
    return metadata


def parse_fasta_header(header: str, metadata: dict):
    """Resolve date and location for a FASTA header via the metadata index.

    Falls back to scraping a year and a location token out of the header when
    the accession is absent from the metadata.
    """
    parts = header.split("|")

    epi_isl = next((p.strip() for p in parts if p.strip().startswith("EPI_ISL_")), None)
    if epi_isl and epi_isl in metadata:
        entry = metadata[epi_isl]
        return entry["date"], entry["location"]

    epi_id = parts[0].strip() if parts else None
    if epi_id and epi_id in metadata:
        entry = metadata[epi_id]
        return entry["date"], entry["location"]

    date = "Unknown"
    location = "Unknown"

    year_match = re.search(r"20\d{2}", header)
    if year_match:
        date = year_match.group()

    if parts and "/" in parts[0]:
        path_parts = parts[0].split("/")
        if len(path_parts) > 1:
            location = path_parts[1]

    return date, location


def stream_fasta(fasta_path, metadata: dict, max_records=None,
                 encoding: str = "latin-1"):
    """Yield ``[date, location, sequence]`` triples from a FASTA file."""
    from Bio import SeqIO

    count = 0
    with open(fasta_path, "r", encoding=encoding, errors="replace") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            date, location = parse_fasta_header(record.description, metadata)
            yield [date, location, str(record.seq)]

            count += 1
            if max_records and count >= max_records:
                break


def load_gisaid(fasta_path, xls_path, max_records=None, encoding: str = "latin-1"):
    """Load GISAID sequences with their metadata into a list of records."""
    metadata = load_gisaid_metadata(xls_path)

    print(f"\nStreaming sequences from {fasta_path}...")
    records = list(tqdm(
        stream_fasta(fasta_path, metadata, max_records=max_records,
                     encoding=encoding),
        desc="Loading FASTA",
    ))
    print(f"Loaded {len(records):,} records")
    return records


def save_records(records, filename="records.npz"):
    """Cache parsed records so the FASTA does not have to be re-streamed."""
    print(f"Saving {len(records):,} records to {filename}...")
    array = np.array(records, dtype=object)
    np.savez_compressed(
        filename,
        dates=array[:, 0],
        locations=array[:, 1],
        sequences=array[:, 2],
    )
    size_mb = Path(filename).stat().st_size / 1024 / 1024
    print(f"Saved {filename} ({size_mb:.2f} MB)")


def load_records(filename="records.npz"):
    print(f"Loading records from {filename}...")
    data = np.load(filename, allow_pickle=True)
    records = [
        [str(d), str(l), str(s)]
        for d, l, s in zip(data["dates"], data["locations"], data["sequences"])
    ]
    print(f"Loaded {len(records):,} records")
    return records
