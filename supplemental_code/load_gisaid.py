import re

import pandas as pd
from Bio import SeqIO
from tqdm import tqdm


def load_gisaid_metadata(xls_path):
    print(f"Loading metadata from {xls_path}...")
    try:
        df = pd.read_excel(xls_path, engine="xlrd")
    except Exception as e:
        print(f"  xlrd engine failed ({e}); retrying with openpyxl...")
        df = pd.read_excel(xls_path, engine="openpyxl")

    df.columns = [str(c).strip() for c in df.columns]
    print(f"Metadata rows: {len(df):,}")

    def find_col(candidates):
        for c in candidates:
            matches = [col for col in df.columns if c.lower() in col.lower()]
            if matches:
                return matches[0]
        return None

    date_col = find_col(["Collection_Date", "Collection Date", "collection_date",
                         "Isolate_Collection_date"])
    location_col = find_col(["Location", "Country", "Geo_Location"])
    epi_isl_col = find_col(["EPI_ISL", "Isolate_Id", "Accession_ID", "Accession ID"])
    epi_id_col = find_col(["Segment_Id", "EPI2", "EPI Sequence Accession"])

    metadata = {}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Indexing metadata"):
        date = str(row[date_col]).strip() if date_col else "Unknown"
        location = str(row[location_col]).strip() if location_col else "Unknown"
        entry = {"date": date, "location": location}
        if epi_isl_col and pd.notna(row[epi_isl_col]):
            metadata[str(row[epi_isl_col]).strip()] = entry
        if epi_id_col and pd.notna(row[epi_id_col]):
            metadata[str(row[epi_id_col]).strip()] = entry

    print(f"Metadata index built: {len(metadata):,} entries")
    return metadata


def parse_fasta_header_with_metadata(header, metadata, segment_field_index=1):
    parts = header.split("|")
    segment = (parts[segment_field_index].strip().upper()
               if len(parts) > segment_field_index else "UNKNOWN")

    epi_isl = next((p.strip() for p in parts if p.strip().startswith("EPI_ISL_")), None)
    if epi_isl and epi_isl in metadata:
        entry = metadata[epi_isl]
        return entry["date"], entry["location"], segment

    epi_id = parts[0].strip() if parts else None
    if epi_id and epi_id in metadata:
        entry = metadata[epi_id]
        return entry["date"], entry["location"], segment

    date, location = "Unknown", "Unknown"
    year_match = re.search(r"(18|19|20)\d{2}", header)
    if year_match:
        date = year_match.group()
    if "/" in parts[0]:
        path_parts = parts[0].split("/")
        if len(path_parts) > 1:
            location = path_parts[1]
    return date, location, segment


def load_gisaid_fasta_with_metadata(fasta_path, metadata, max_records=None, encoding="latin-1"):
    count = 0
    with open(fasta_path, "r", encoding=encoding, errors="replace") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            date, location, segment = parse_fasta_header_with_metadata(record.description, metadata)
            yield [date, location, str(record.seq), segment]
            count += 1
            if max_records and count >= max_records:
                break


def load_gisaid(fasta_path, xls_path, max_records=None, encoding="latin-1", segment_filter=None):
    metadata = load_gisaid_metadata(xls_path)

    print(f"\nStreaming sequences from {fasta_path}...")
    raw_records = list(tqdm(
        load_gisaid_fasta_with_metadata(fasta_path, metadata, max_records=max_records,
                                        encoding=encoding),
        desc="Loading FASTA",
    ))
    print(f"Loaded {len(raw_records):,} records")

    seg_counts = {}
    for *_, seg in raw_records:
        seg_counts[seg] = seg_counts.get(seg, 0) + 1
    print("Segment breakdown:", dict(sorted(seg_counts.items(), key=lambda kv: -kv[1])))
    if segment_filter is None and len(seg_counts) > 1:
        print("WARNING: multiple gene segments detected and no segment_filter applied.")

    if segment_filter is not None:
        records = [[d, l, s] for d, l, s, seg in raw_records if seg == segment_filter.upper()]
        print(f"Kept {len(records):,}/{len(raw_records):,} records for segment '{segment_filter}'")
    else:
        records = [[d, l, s] for d, l, s, _ in raw_records]
    return records
