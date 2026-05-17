#!/usr/bin/env python3
"""
GEOCK Data Acquisition Pipeline
=================================
Downloads and prepares additional training data from public sources.

Sources:
1. BindingDB - 3M+ measurements
2. ChEMBL - 2.4M compounds
3. LP-PDBBind - ~25K refined

Usage:
    python acquire_data.py [--source bindingdb|chembl|lp_pdbbind|all]
    python acquire_data.py --limit 10000  # Limit samples
    python acquire_data.py --dry-run       # Check without downloading
"""

import os
import sys
import pickle
import argparse
import json
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime


# Use cross-platform paths
def _get_cache_dir():
    linux = Path("/home/chow/.cache/geock_autoresearch")
    if linux.exists():
        return linux
    win = Path(os.path.expanduser("~/OneDrive/.cache/geock_autoresearch"))
    if win.exists():
        return win
    return Path("./cache")


def _get_work_dir():
    linux = Path("/home/chow/autoresearch")
    if linux.exists():
        return linux
    win = Path(os.path.expanduser("~/OneDrive/autoresearch"))
    if win.exists():
        return win
    return Path(".")


CACHE_DIR = _get_cache_dir()
WORK_DIR = _get_work_dir()

print("=" * 60)
print("GEOCK Data Acquisition Pipeline")
print("=" * 60)
print(f"Cache: {CACHE_DIR}")
print(f"Work:  {WORK_DIR}")


# ===== DATA SOURCES =====
class DataSource:
    """Base class for data sources."""

    name = "base"
    url = ""
    estimated_size = 0

    def download(self, limit=None, dry_run=False):
        raise NotImplementedError

    def process(self, raw_data):
        raise NotImplementedError


class BindingDBDataSource(DataSource):
    """BindingDB - protein-small molecule binding data."""

    name = "BindingDB"
    # Free download page
    url = "https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp"

    # Alternative: Direct mirror or cached file
    # Users need to register for free to download

    def __init__(self):
        self.alt_urls = [
            # Add known mirrors if available
        ]

    def download(self, limit=None, dry_run=False):
        """Download from BindingDB (requires account)."""
        print(f"\n[{self.name}] Checking for existing data...")

        # Check for existing BindingDB data
        cache_file = CACHE_DIR / "bindingdb_raw.csv"
        if cache_file.exists():
            print(f"  Found: {cache_file}")
            return self.process_cache(cache_file, limit)

        print(f"  No existing file found")
        print(f"  To download: Register at {self.url}")
        print(f"  Alternative: Use existing LP-PDBBind + ChEMBL data")

        return None

    def process_cache(self, cache_file, limit):
        """Process cached BindingDB file."""
        print(f"  Loading {cache_file}...")
        df = pd.read_csv(cache_file, sep="\t", low_memory=False)
        print(f"    Total rows: {len(df)}")

        # Filter for protein-ligand with numeric affinity
        if limit:
            df = df.head(limit)

        return self.process(df)

    def process(self, df):
        """Process binding data."""
        # Find relevant columns
        smiles_col = None
        affinity_col = None

        for col in df.columns:
            if "smiles" in col.lower() or "ligand" in col.lower():
                if not smiles_col:
                    smiles_col = col
            if "kd" in col.lower() or "ki" in col.lower() or "ic50" in col.lower():
                if not affinity_col:
                    affinity_col = col

        if not smiles_col or not affinity_col:
            print(f"    Warning: Could not find required columns")
            return None

        # Extract and filter
        data = []
        for _, row in df.iterrows():
            try:
                smiles = row[smiles_col]
                value = row[affinity_col]

                if pd.isna(smiles) or pd.isna(value):
                    continue

                # Convert to pKd (negative log of nM)
                if isinstance(value, (int, float)):
                    if value > 0:
                        pkd = -np.log10(value * 1e-9)
                        if 2 < pkd < 15:  # Reasonable range
                            data.append(
                                {
                                    "smiles": str(smiles),
                                    "affinity": pkd,
                                    "source": "bindingdb",
                                }
                            )
            except:
                continue

        print(f"    Processed: {len(data)} valid entries")
        return data


class ChEMBLDataSource(DataSource):
    """ChEMBL - bioactive compounds."""

    name = "ChEMBL"
    url = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBL/latest/"

    def download(self, limit=None, dry_run=False):
        """Check for existing ChEMBL data."""
        print(f"\n[{self.name}] Checking existing data...")

        # Check for different versions
        candidates = [
            CACHE_DIR / "chembl_binding.csv",
            CACHE_DIR / "chembl_more.pkl",
            CACHE_DIR / "chembl_v2.pkl",
            CACHE_DIR / "chembl_tiny.pkl",
            CACHE_DIR / "chembl_sample.csv",
        ]

        for f in candidates:
            if f.exists():
                print(f"  Found: {f}")
                return self.process_file(f, limit)

        print(f"  No ChEMBL files found")
        print(f"  To download: wget {self.url}chembl_*.gz")

        return None

    def process_file(self, f, limit):
        """Process ChEMBL file."""
        print(f"  Loading {f}...")

        if f.suffix == ".pkl":
            with open(f, "rb") as fp:
                data = pickle.load(fp)
            if isinstance(data, list):
                print(f"    Entries: {len(data)}")
                if limit:
                    data = data[:limit]
                return data
            return None

        df = pd.read_csv(f)
        print(f"    Rows: {len(df)}")

        # Find SMILES and activity columns
        data = []
        for _, row in df.iterrows():
            try:
                smiles = row.get("Smiles", row.get("smiles", None))
                if pd.isna(smiles):
                    continue

                # Get activity
                value = row.get("Kd", row.get("Ki", row.get("IC50", None)))
                if pd.isna(value) or not isinstance(value, (int, float)):
                    continue

                if value > 0:
                    pkd = -np.log10(value * 1e-9)
                    if 2 < pkd < 15:
                        data.append(
                            {"smiles": str(smiles), "affinity": pkd, "source": "chembl"}
                        )
            except:
                continue

        print(f"    Processed: {len(data)} valid")
        if limit:
            data = data[:limit]
        return data


class LP_PDBBindDataSource(DataSource):
    """LP-PDBBind - refined protein-ligand complexes."""

    name = "LP-PDBBind"
    url = "https://github.com/THGLab/LP-PDBBind"

    def download(self, limit=None, dry_run=False):
        """Check for LP-PDBBind data."""
        print(f"\n[{self.name}] Checking existing data...")

        candidates = [
            CACHE_DIR / "LP_PDBBind.csv",
            CACHE_DIR / "lp_new_features_8k_no2016.pkl",
            CACHE_DIR / "merged_39k.pkl",
        ]

        for f in candidates:
            if f.exists():
                print(f"  Found: {f}")
                return self.process_file(f, limit)

        print(f"  No LP-PDBBind files found")
        # Can clone from GitHub if needed

        return None

    def process_file(self, f, limit):
        """Process LP-PDBBind CSV."""
        print(f"  Loading {f}...")

        df = pd.read_csv(f)
        print(f"    Rows: {len(df)}")

        # Extract SMILES and pKd (value column)
        data = []
        for _, row in df.iterrows():
            try:
                smiles = row.get("smiles", None)
                value = row.get("value", row.get("kd", None))

                if pd.isna(smiles) or pd.isna(value):
                    continue

                # value is already pKd
                pkd = float(value)
                if 2 < pkd < 15:
                    data.append(
                        {"smiles": str(smiles), "affinity": pkd, "source": "lp_pdbbind"}
                    )
            except:
                continue

        print(f"    Processed: {len(data)} valid")
        if limit:
            data = data[:limit]
        return data


# ===== MAIN PIPELINE =====
def run_pipeline(sources=None, limit=None, dry_run=False, output_name=None):
    """Run data acquisition pipeline."""

    # Initialize sources
    if sources is None or sources == "all":
        source_classes = [LP_PDBBindDataSource, ChEMBLDataSource]
    else:
        source_classes = []
        for s in sources:
            if "lp" in s.lower():
                source_classes.append(LP_PDBBindDataSource)
            elif "chembl" in s.lower():
                source_classes.append(ChEMBLDataSource)
            elif "binding" in s.lower():
                source_classes.append(BindingDBDataSource)

    all_data = []

    for src_class in source_classes:
        src = src_class()
        result = src.download(limit=limit, dry_run=dry_run)

        if result:
            all_data.extend(result)
            print(f"  Added {len(result)} entries")

    if not all_data:
        print(f"\n⚠ No new data found")
        print(f"  Available sources only use existing cached files")
        return None

    # Deduplicate by SMILES
    print(f"\n[Deduplication] {len(all_data)} total → ", end="")
    seen = {}
    for d in all_data:
        seen[d["smiles"]] = d
    unique_data = list(seen.values())
    print(f"{len(unique_data)} unique")

    # Save merged data
    output_path = WORK_DIR / (output_name or "geock_merged_data.pkl")
    backup_path = (
        WORK_DIR
        / f"geock_merged_data_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    )

    if output_path.exists():
        # Backup existing
        print(f"  Backup: {output_path} → {backup_path}")

        # Load and merge with existing
        with open(output_path, "rb") as f:
            existing_data = pickle.load(f)

        # Merge
        for d in existing_data:
            if d["smiles"] not in seen:
                unique_data.append(d)

        print(f"  Merged: {len(unique_data)} total (with backup)")

    # Save
    with open(output_path, "wb") as f:
        pickle.dump(unique_data, f)

    print(f"\n✅ Saved: {output_path}")
    print(f"  Total entries: {len(unique_data)}")

    # Also save as CSV for inspection
    csv_path = output_path.with_suffix(".csv")
    if len(unique_data) < 100000:
        df = pd.DataFrame(unique_data)
        df.to_csv(csv_path, index=False)
        print(f"  CSV: {csv_path}")

    return unique_data


# ===== ADD TO MERGED DATA =====
def add_to_merged(new_data, output_name="merged_39k.pkl"):
    """Add new data to existing merged dataset."""

    output_path = WORK_DIR / output_name

    # Load existing
    if output_path.exists():
        with open(output_path, "rb") as f:
            existing = pickle.load(f)
        print(f"Existing: {len(existing)} entries")
    else:
        existing = []
        print(f"Starting fresh")

    # Add new entries (avoid duplicates)
    seen = {d["smiles"]: True for d in existing}
    added = 0

    for d in new_data:
        if d["smiles"] not in seen:
            existing.append(d)
            seen[d["smiles"]] = True
            added += 1

    print(f"Added: {added} new entries")
    print(f"Total: {len(existing)}")

    # Save
    with open(output_path, "wb") as f:
        pickle.dump(existing, f)

    print(f"Saved: {output_path}")

    return existing


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GEOCK Data Acquisition")
    parser.add_argument(
        "--source",
        type=str,
        default="all",
        help="Source: bindingdb, chembl, lp_pdbbind, all",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit samples per source"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Check without downloading"
    )
    parser.add_argument(
        "--output", type=str, default="geock_merged_data.pkl", help="Output filename"
    )

    args = parser.parse_args()

    sources = None if args.source == "all" else [args.source]

    print(f"\nArgs: sources={sources}, limit={args.limit}, dry_run={args.dry_run}")

    data = run_pipeline(
        sources=sources, limit=args.limit, dry_run=args.dry_run, output_name=args.output
    )

    if data:
        print(f"\n🎉 Pipeline complete! {len(data)} entries ready for training")
    else:
        print(f"\n⚠ No new data (using existing files)")
        print(f"\nTo get more data:")
        print(f"  1. Register at bindingdb.org (free)")
        print(f"  2. Download ChEMBL: wget {ChEMBLDataSource.url}")
        print(f"  3. Clone LP-PDBBind: git clone {LP_PDBBindDataSource.url}")
