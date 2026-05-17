"""
fetch_casf.py — CASF-2016 Benchmark Data Fetcher
================================================
CASF-2016 is the gold-standard scoring function benchmark:
  - 285 high-quality protein-ligand complexes
  - All with experimental binding data
  - Pre-docked poses (no docking needed)
  - Publicly downloadable

Download from: Zhang et al. / CASF-2016 on Figshare
https://doi.org/10.6084/m9.figshare.12368363

Usage:
  python fetch_casf.py              # download and extract
  python extract_casf.py            # compute features
"""

import os
import sys
import json
import tarfile
import zipfile
import shutil
import hashlib
import requests
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import gdown
    GDRIVE_OK = True
except ImportError:
    GDRIVE_OK = False

OUT_DIR  = Path("/mnt/c/Users/yakka/Downloads/geock_casf_data")
CASF_DIR = OUT_DIR / "CASF2016"
INDEX_URL = "https://raw.githubusercontent.com/yuke3/DeepRMS/master/data/casf2016_index.json"
FIGSHARE_URL = "https://figshare.com/ndownloader/files/"

FIGSHARE_ID = "34689761"  # CASF-2016 complete dataset


def log(msg: str):
    print(f"  {msg}", flush=True)


def download_casf(limit_gb: float = 2.0) -> bool:
    """
    Download CASF-2016 from Figshare.
    The dataset has ~1.5 GB of PDB structures + data files.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CASF_DIR.mkdir(exist_ok=True)

    # Check if already downloaded
    if (CASF_DIR / "INDEX").exists() and (CASF_DIR / "Crystal Structures").exists():
        log("CASF-2016 already downloaded")
        return True

    log(f"Downloading CASF-2016 from Figshare (~1.5 GB)...")
    log("This may take 10-30 minutes depending on your connection.")

    # Try direct Figshare download
    tar_path = OUT_DIR / "casf2016.tar.gz"
    url = f"{FIGSHARE_URL}{FIGSHARE_ID}"

    try:
        log(f"Fetching: {url}")
        with requests.get(url, stream=True, timeout=300, headers={
            "User-Agent": "GEOCK-research/2.0"
        }) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            log(f"Size: {total / 1e9:.1f} GB")
            downloaded = 0
            chunk_size = 1024 * 1024
            with open(tar_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0 and downloaded % (100 * chunk_size) == 0:
                            log(f"  {downloaded/total*100:.0f}%  {downloaded/1e6:.0f}MB")
            log(f"Downloaded: {downloaded/1e6:.0f}MB")

    except Exception as e:
        warn(f"Figshare download failed: {e}")
        log("Trying alternative: download individual PDB files from RCSB...")
        return False

    # Extract
    try:
        log("Extracting...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(OUT_DIR)
        log("Extracted successfully")
        tar_path.unlink(missing_ok=True)
        return True
    except Exception as e:
        warn(f"Extraction failed: {e}")
        return False


def parse_casf_index() -> list[dict]:
    """Parse CASF-2016 INDEX file into compounds."""
    index_file = CASF_DIR / "INDEX"
    if not index_file.exists():
        log(f"No INDEX file at {index_file}")
        return []

    compounds = []
    with open(index_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Format: PDB    Year   Resolution   Ki(uM)   -logKi  LigName
            parts = line.split()
            if len(parts) < 5:
                continue

            pdb_id = parts[0].lower()
            year   = int(parts[1]) if parts[1].isdigit() else 2010
            res    = float(parts[2]) if parts[2].replace('.','',1).isdigit() else 2.0
            ki_str = parts[3] if len(parts) > 3 else ""
            ki_val = float(ki_str) if ki_str.replace('.','',1).replace('e','',1).replace('-','',1).replace('+','',1).isdigit() else 0
            ki_unit = parts[4] if len(parts) > 4 else "nM"

            # Convert Ki to ΔG
            RT = 0.593
            if ki_unit.lower() == "um" or "μm" in ki_unit:
                ki_molar = ki_val * 1e-6
            elif ki_unit.lower() == "mm":
                ki_molar = ki_val * 1e-3
            elif ki_unit.lower() == "pm":
                ki_molar = ki_val * 1e-12
            else:  # nM
                ki_molar = ki_val * 1e-9

            if ki_molar <= 0:
                continue

            delta_g = RT * (-1) * (22.0 - np.log10(ki_molar) * 2.303)
            delta_g = round(RT * (-1) * np.log(ki_molar), 3)

            if delta_g > 0 or delta_g < -25:
                continue

            # Find ligand SMILES from CASF ligand file
            smiles = get_casf_smiles(pdb_id)

            compounds.append({
                "pdb_id": pdb_id,
                "smiles": smiles,
                "experimental_affinity": delta_g,
                "affinity_type": "Ki",
                "year": year,
                "resolution": res,
                "source": "CASF-2016",
                "pocket_file": str(CASF_DIR / "Crystal Structures" / pdb_id / f"{pdb_id}_ligand.mol2"),
                "protein_file": str(CASF_DIR / "Crystal Structures" / pdb_id / f"{pdb_id}_protein.pdb"),
            })

    log(f"Parsed {len(compounds)} CASF-2016 complexes")
    return compounds


def get_casf_smiles(pdb_id: str) -> str:
    """Get SMILES for CASF ligand from the ligand file."""
    lig_dir = CASF_DIR / "Crystal Structures" / pdb_id
    mol2_file = lig_dir / f"{pdb_id}_ligand.mol2"

    if mol2_file.exists():
        try:
            with open(mol2_file) as f:
                content = f.read()
            # Extract SMILES from mol2 comment
            for line in content.splitlines():
                if "SMILES" in line or "Name" in line:
                    parts = line.split()
                    for p in parts:
                        if p.startswith("SMILES"):
                            continue
                        try:
                            from rdkit import Chem
                            mol = Chem.MolFromSmiles(p)
                            if mol:
                                return p
                        except:
                            pass
        except:
            pass
    return ""


def main():
    print("=" * 60)
    print("  CASF-2016 Data Fetcher")
    print("=" * 60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Try to download
    ok = download_casf()
    if not ok:
        log("Automatic download failed.")
        log("Please download CASF-2016 manually from:")
        log("  https://doi.org/10.6084/m9.figshare.12368363")
        log("  Extract to: " + str(OUT_DIR))
        log("Then run: python fetch_casf.py --parse-only")
        return

    # Parse INDEX
    compounds = parse_casf_index()
    if not compounds:
        return

    # Save
    out_file = OUT_DIR / "compounds_casf.json"
    with open(out_file, "w") as f:
        json.dump(compounds, f, indent=2)

    print(f"\n[OK] Saved {len(compounds)} CASF-2016 complexes")
    print(f"     {out_file}")
    print(f"\n  Next: python extract_casf.py")


if __name__ == "__main__":
    main()

# numpy import for the delta_g calculation
import numpy as np
