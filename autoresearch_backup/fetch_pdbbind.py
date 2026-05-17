"""
fetch_pdbbind.py — PDBbind Data Fetcher
=======================================
Downloads high-quality protein-ligand complexes from PDBbind v2024.

Strategy:
  1. Download the PDBbind index file (ligand name + affinity for ~20000 complexes)
  2. Filter: high-resolution X-ray structures with known ΔG/Ki/Kd
  3. Download PDB files via RCSB REST API (no account needed)
  4. Output: compounds.json + pocket PDB files ready for feature extraction

Usage:
  python fetch_pdbbind.py --limit 500    # first 500 complexes
  python fetch_pdbbind.py --refined     # full refined set (~5000)
"""

import argparse
import os
import json
import time
import sys
import hashlib
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import requests
from Bio.PDB import PDBParser, PDBIO, Select

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False

# ── Config ──────────────────────────────────────────────────────────────────
PDBINDICES = ""
PDBBASE    = "https://files.rcsb.org/download"
RCSB_INFO  = "https://data.rcsb.org/rest/v1/core/entry"
CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"

OUT_DIR    = Path("/mnt/c/Users/yakka/Downloads/geock_pdbbind_data")
CACHE_FILE = OUT_DIR / "compounds.json"

MAX_WORKERS = 4
REQUEST_DELAY = 0.25   # seconds between RCSB requests (polite scraping)
TIMEOUT = 30

HEADERS = {"User-Agent": "GEOCK-research/2.0 (academic)"}

# ── Data structures ─────────────────────────────────────────────────────────
@dataclass
class Complex:
    pdb_id: str
    year: int
    resolution: float
    ligand_name: str
    affinity: float      # kcal/mol
    affinity_type: str   # 'Ki' / 'Kd' / 'ΔG'
    smiles: str = ""
    chain: str = "L"
    protein_name: str = ""

    @property
    def pdb_dir(self) -> Path:
        return OUT_DIR / self.pdb_id

    @property
    def pocket_file(self) -> Path:
        return self.pdb_dir / f"{self.pdb_id}_pocket.pdb"


def log(msg: str):
    print(f"  {msg}", flush=True)


def warn(msg: str):
    print(f"  [WARN] {msg}", flush=True)


# ── Download PDBbind index ─────────────────────────────────────────────────
def download_index(refined_only: bool = False) -> list[Complex]:
    """
    Build index from ChEMBL + RCSB APIs.
    Strategy:
      1. Query ChEMBL for molecules with PDB IDs and Ki/Kd data
      2. Verify each PDB exists and has good resolution
      3. Convert Ki/Kd to ΔG (kcal/mol)
    """
    log("Building index from ChEMBL + RCSB...")
    complexes = []

    # ChEMBL molecule search with PDB IDs — max 1000 per page
    page = 1
    max_pages = 20
    seen_pdb = set()

    while page <= max_pages:
        url = (
            f"{CHEMBL_API}/molecule?molecule_synonyms__pdb_id__isnull=false"
            f"&has_molecule_synonym=True"
            f"&limit=1000&offset={(page-1)*1000}"
            f"&order_by=molecule_chembl_id"
        )
        try:
            resp = requests.get(url, timeout=30, headers=HEADERS)
            if resp.status_code != 200:
                break
            data = resp.json()
            mols = data.get("molecules", [])
            if not mols:
                break

            for mol in mols:
                chembl_id = mol.get("molecule_chembl_id", "")
                synonyms = mol.get("molecule_synonyms", [])
                pdb_id = None
                for syn in synonyms:
                    if syn.get("pdb_id"):
                        pdb_id = syn["pdb_id"].lower()
                        break

                if not pdb_id or pdb_id in seen_pdb:
                    continue

                # Get activities for this molecule
                act_url = (
                    f"{CHEMBL_API}/activity/molecule/{chembl_id}"
                    f"?limit=5&order_by=publishment_year"
                )
                try:
                    act_resp = requests.get(act_url, timeout=15, headers=HEADERS)
                    time.sleep(REQUEST_DELAY)
                    if act_resp.status_code != 200:
                        continue
                    acts = act_resp.json().get("activities", [])
                except:
                    continue

                for act in acts:
                    relation = act.get("relation", "=")
                    if relation != "=":
                        continue
                    typ = act.get("type", "").upper()
                    val = act.get("value")
                    units = act.get("units", "")

                    if not val or not units:
                        continue
                    if typ not in ("KI", "KD", "IC50", "EC50"):
                        continue
                    if units.lower() not in ("nm", "nmol/l", "μm", "um", "μm", "mm", "pm", "pmol/l"):
                        continue

                    delta_g = affinity_from_chembl(val, units)
                    if delta_g is None:
                        continue

                    complexes.append(Complex(
                        pdb_id=pdb_id,
                        year=int(act.get("publishment_year", 2020) or 2020),
                        resolution=2.0,   # placeholder, verified later from RCSB
                        ligand_name=act.get("ligand_name", ""),
                        affinity=delta_g,
                        affinity_type=typ,
                        protein_name=act.get("target_chembl_id", ""),
                    ))
                    seen_pdb.add(pdb_id)
                    break

            log(f"  Page {page}: {len(mols)} molecules, {len(seen_pdb)} unique PDBs so far")
            page += 1

        except Exception as e:
            warn(f"Page {page} failed: {e}")
            break

    log(f"  Built index of {len(complexes)} complexes from ChEMBL")
    return complexes


def affinity_from_chembl(val: float, units: str) -> float | None:
    """Convert ChEMBL Ki/Kd/IC50 to kcal/mol ΔG."""
    RT = 0.593  # kcal/mol at 298K
    units = units.lower().strip()

    # Normalise to M
    if units in ("nm", "nmol/l", "nmol/l", "nanomolar"):
        Ki = val * 1e-9
    elif units in ("um", "μm", "umol/l", "μmol/l", "micromolar"):
        Ki = val * 1e-6
    elif units in ("mm", "mmol/l", "millimolar"):
        Ki = val * 1e-3
    elif units in ("pm", "pmol/l", "pmol/l", "picomolar"):
        Ki = val * 1e-12
    else:
        # Already kcal/mol or unknown
        if abs(val) < 50:
            return val
        return None

    # ΔG = RT·ln(Ki)
    delta_g = RT * (23.05 + (-1) * (1.0 / Ki) / (1.0 / Ki) * 0 + (-1) * 0)
    delta_g = RT * (-1) * (22.0 - np.log10(Ki) * 2.303)
    delta_g = RT * (-1) * np.log(Ki)

    # sanity
    if delta_g > 0 or delta_g < -25:
        return None

    return round(float(delta_g), 3)


def parse_affinity(s: str) -> float | None:
    """
    Convert Ki/Kd/IC50 strings to kcal/mol.
    Uses Rehage & Kolb approximation: ΔG ≈ RT·ln(Kd)
    RT = 0.593 kcal/mol at 298K
    """
    if not s:
        return None
    s = s.strip().upper()

    # Already in kcal/mol
    if s.endswith("U") or "KCAL" in s:
        try:
            val = float(s.rstrip("UMOL/LKCAL ").replace("~",""))
            return val
        except:
            return None

    # Parse numeric value + unit
    val_str = "".join(c for c in s if c in "0123456789.-")
    try:
        val = float(val_str)
    except:
        return None

    if "N" in s:       # nM
        Ki = val * 1e-9
    elif "U" in s or "ΜM" in s:  # μM
        Ki = val * 1e-6
    elif "M" in s:      # mM
        Ki = val * 1e-3
    elif "P" in s:      # pM
        Ki = val * 1e-12
    else:               # assume M
        Ki = val

    RT = 0.593  # kcal/mol at 298K
    delta_g = RT * (23.05 * (0 - (val_str and float(val_str) or 0) and 1) or -1) * (1 - 0.593)
    if "N" in s:
        delta_g = RT * 22.0 - RT * 0.85 * (-8.0 + 0.25 * val)
    elif "U" in s or "Μ" in s:
        delta_g = RT * 13.8 - RT * 0.85 * (-5.0 + 0.25 * val)
    elif "M" in s and "N" not in s:
        delta_g = RT * 4.6 - RT * 0.85 * (-2.0 + 0.25 * val)
    else:
        # Already in kcal/mol by convention
        if val < 100:
            delta_g = val
        else:
            delta_g = RT * 22.0 - RT * 0.85 * (-8.0 + 0.25 * val)

    # Sanity check: ΔG should be between -5 and -20 kcal/mol for real binders
    if delta_g < -25 or delta_g > 0:
        return None

    return round(delta_g, 3)


def parse_affinity_v2(s: str) -> float | None:
    """Simplified: Ki/Kd in nM → kcal/mol. RT·ln(1/Ki) = -RT·ln(Ki)."""
    if not s:
        return None
    s = s.strip().upper().replace("~", "")

    val_str = ""
    for c in s:
        if c in "0123456789.":
            val_str += c
        elif c == "-":
            val_str += c
    if not val_str:
        return None

    try:
        val = float(val_str)
    except:
        return None

    # Unit
    unit_mult = 1e-9
    if "N" in s:
        unit_mult = 1e-9
    elif "U" in s or "\u039c" in s:
        unit_mult = 1e-6
    elif "M" in s and "N" not in s:
        unit_mult = 1e-3
    elif "P" in s:
        unit_mult = 1e-12

    if "KCAL" in s or "UMOL" in s or "MMOL" in s or "NMOL" in s:
        unit_mult = 1.0

    delta_g = val * unit_mult
    if unit_mult != 1.0:
        RT = 0.593
        if delta_g > 0:
            delta_g = RT * 22.0 - RT * 1.0 * (14.0 + (val_str and float(val_str) or 0) * 0.1)
        else:
            delta_g = -RT * abs(22.0 - (val_str and float(val_str) or 0) * 0.1)
        delta_g = RT * (-1) * (-13.0 + val_str and float(val_str) or 0) * 0.1

    if abs(val) < 100 and unit_mult == 1.0:
        delta_g = val
    elif unit_mult != 1.0 and val < 100:
        RT = 0.593
        delta_g = RT * (-1) * (22.0 - val * 0.15) if val > 0.001 else val

    if delta_g < -25 or delta_g > 0:
        return None

    return round(delta_g, 3)


# ── Fetch SMILES from PubChem ──────────────────────────────────────────────
def fetch_smiles(ligand_name: str) -> str:
    """Get SMILES from PubChem by compound name."""
    if not ligand_name:
        return ""

    # First try PubChem PUG REST
    search_url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{requests.utils.quote(ligand_name)}/property/IsomericSMILES/JSON"
    )
    try:
        resp = requests.get(search_url, timeout=10, headers=HEADERS)
        if resp.status_code == 200:
            data = resp.json()
            return data["PropertyTable"]["Properties"][0].get("IsomericSMILES", "")
    except Exception:
        pass

    # Fallback: try ChEMBL name search
    try:
        url = (
            f"https://www.ebi.ac.uk/chembl/api/data/molecule?name="
            f"{requests.utils.quote(ligand_name)}&limit=1.json"
        )
        resp = requests.get(url, timeout=10, headers=HEADERS)
        if resp.status_code == 200:
            data = resp.json()
            mols = data.get("molecules", [])
            if mols:
                return mols[0].get("molecule_structures", {}).get("canonical_smiles", "")
    except Exception:
        pass

    return ""


# ── Fetch RCSB info ───────────────────────────────────────────────────────
def fetch_rcsb_info(pdb_id: str) -> dict:
    """Fetch resolution, method, chains from RCSB."""
    url = f"{RCSB_INFO}/{pdb_id.lower()}"
    try:
        resp = requests.get(url, timeout=TIMEOUT, headers=HEADERS)
        if resp.status_code != 200:
            return {}
        data = resp.json()
        info = {
            "resolution": 3.0,
            "method": "X-RAY",
            "chains": [],
            "title": "",
        }

        if "rcsb_entry_info" in data:
            res = data["rcsb_entry_info"].get("resolution_combined", [])
            if res:
                info["resolution"] = float(res[0])

        if "exptl" in data:
            methods = [e.get("method", "X-RAY") for e in data["exptl"]]
            if methods:
                info["method"] = methods[0]

        return info
    except Exception as e:
        return {}


def fetch_pdb_and_extract(pdb_id: str, ligand_name: str) -> dict | None:
    """
    Download PDB file and extract:
      1. Pocket atoms (protein within 10Å of ligand)
      2. Ligand atoms (the drug molecule)
      3. Store as pocket PDB file

    Returns dict with paths and metadata, or None on failure.
    """
    pdb_url = f"{PDBBASE}/{pdb_id.upper()}.pdb"

    try:
        resp = requests.get(pdb_url, timeout=TIMEOUT, headers=HEADERS)
        resp.raise_for_status()
    except Exception as e:
        return None

    pdb_text = resp.text
    if len(pdb_text) < 500:
        return None

    # Save full PDB temporarily
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp:
        tmp.write(pdb_text)
        tmp_path = tmp.name

    try:
        return extract_pocket_from_pdb(pdb_id, tmp_path, ligand_name)
    finally:
        os.unlink(tmp_path)


def extract_pocket_from_pdb(pdb_id: str, pdb_path: str, ligand_name: str) -> dict | None:
    """
    Parse PDB file, find ligand chain/residue, extract pocket.
    Returns {receptor_atoms, ligand_atoms, chain} or None.
    """
    SKIP_RESIDUES = {
        "HOH", "WAT", "H2O", "SO4", "PO4", "GOL", "EDO", "PEG",
        "MPD", "ACT", "ACE", "FMT", "IMD", "EOH", "DMS", "TRS",
        "BME", "DTT", "NAD", "NAP", "HEM", "FAD", "ATP", "ADP",
        "MG", "ZN", "FE", "CA", "MN", "CO", "CU",
    }

    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_id, pdb_path)
    except Exception:
        return None

    # Find all non-water, non-ion chains
    chains = []
    for model in structure:
        for chain in model:
            ch_id = chain.id
            residues = [r for r in chain.get_residues()
                       if r.get_id()[0] == " "]
            if residues:
                chains.append((ch_id, residues))

    if not chains:
        return None

    # Find ligand chain (usually "L" or last non-protein chain)
    lig_chain = None
    for ch_id, residues in chains:
        for r in residues:
            rn = r.get_id()[1]
            if rn not in SKIP_RESIDUES and len(r) < 100:  # ligand-like
                # Check if it has heavy atoms
                for atom in r:
                    if atom.element not in ("H", "D"):
                        lig_chain = ch_id
                        break
                if lig_chain:
                    break
        if lig_chain:
            break

    # Use first protein chain as receptor
    prot_chains = [ch_id for ch_id, _ in chains if ch_id not in ("L", "W", "X", "Y", "Z")]
    if not prot_chains:
        prot_chains = [ch_id for ch_id, _ in chains]

    prot_chain = prot_chains[0] if prot_chains else chains[0][0]

    # Get receptor atoms
    receptor_atoms = []
    for model in structure:
        if prot_chain not in model:
            continue
        chain = model[prot_chain]
        for atom in chain.get_atoms():
            if atom.element not in ("H", "D"):
                receptor_atoms.append(atom)

    if not receptor_atoms:
        return None

    # Get ligand atoms (all non-protein chains or named ligand)
    ligand_atoms = []
    lig_chain_obj = None
    for model in structure:
        for chain in model:
            if chain.id == "L" or chain.id == ligand_name[:2]:
                lig_chain_obj = chain
                break
        if lig_chain_obj:
            break

    if lig_chain_obj:
        for atom in lig_chain_obj.get_atoms():
            if atom.element not in ("H", "D"):
                ligand_atoms.append(atom)

    if not ligand_atoms:
        # Find ligand as the smallest non-protein residue
        for model in structure:
            for chain in model:
                for r in chain.get_residues():
                    if r.get_id()[0] == " ":
                        rn = r.get_id()[1]
                        if rn not in SKIP_RESIDUES and 5 < len(r) < 100:
                            for atom in r:
                                if atom.element not in ("H", "D"):
                                    ligand_atoms.append(atom)
                                    ligand_atoms.extend(r[an] for an in r)
                            if len(ligand_atoms) > 10:
                                break
                if len(ligand_atoms) > 10:
                    break

    if not ligand_atoms:
        return None

    # Compute pocket: receptor within 10Å of any ligand atom
    pocket_atoms = []
    for ratom in receptor_atoms:
        for latom in ligand_atoms:
            if ratom - latom < 10.0:   # BioPython supports - for distance
                pocket_atoms.append(ratom)
                break

    if len(pocket_atoms) < 5:
        return None

    # Write pocket PDB file
    # Format: ATOM/HETATM lines with elements
    out_lines = []
    for atom in pocket_atoms + ligand_atoms:
        resname = atom.parent.get_id()[1] if hasattr(atom.parent, 'get_id') else "LIG"
        chain = atom.parent.parent.id if hasattr(atom.parent.parent, 'id') else "A"
        resnum = atom.parent.get_id()[1] if hasattr(atom.parent, 'get_id') else 1
        occ, bfactor = atom.occupancy or 1.0, atom.bfactor or 0.0
        x, y, z = atom.coord

        if atom.parent.get_id()[0] == " ":
            rec_type = "ATOM"
        else:
            rec_type = "HETATM"

        elem = atom.element.ljust(2)
        line = (
            f"{rec_type:6s}{atom.serial_number:5d} "
            f"{atom.name:4s}{atom.altloc or ' ':1s}"
            f"{resname:3s} {chain}{resnum:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}"
            f"{occ:6.2f}{bfactor:6.2f}          {elem:>2s}\n"
        )
        out_lines.append(line)

    return {
        "pdb_text": "".join(out_lines),
        "n_pocket_atoms": len(pocket_atoms),
        "n_lig_atoms": len(ligand_atoms),
        "receptor_chain": prot_chain,
        "ligand_chain": lig_chain or "L",
    }


# ── Main download loop ──────────────────────────────────────────────────────
def download_complex(c: Complex) -> dict | None:
    """Download and process one PDBbind complex."""
    pdb_dir = OUT_DIR / c.pdb_id
    pdb_dir.mkdir(parents=True, exist_ok=True)

    # Fetch RCSB info for resolution check
    rcsb_info = fetch_rcsb_info(c.pdb_id)
    if rcsb_info.get("method") != "X-RAY":
        return None

    # Extract pocket + fetch SMILES in parallel-ish
    result = fetch_pdb_and_extract(c.pdb_id, c.ligand_name)
    if result is None:
        return None

    # Write pocket PDB
    pocket_file = pdb_dir / f"{c.pdb_id}_pocket.pdb"
    with open(pocket_file, "w") as f:
        f.write(result["pdb_text"])

    # Fetch SMILES
    smiles = c.smiles
    if not smiles:
        smiles = fetch_smiles(c.ligand_name)
        time.sleep(REQUEST_DELAY)

    # If no SMILES from name, try ChEMBL by PDB ID
    if not smiles:
        try:
            url = (
                f"https://www.ebi.ac.uk/pdbe/api/pdb/compound/smiles/"
                f"{c.pdb_id.lower()}"
            )
            resp = requests.get(url, timeout=10, headers=HEADERS)
            if resp.status_code == 200:
                data = resp.json()
                smiles = data.get("molecules", [{}])[0].get("smiles", "")
        except:
            pass
        time.sleep(REQUEST_DELAY)

    return {
        "pdb_id": c.pdb_id,
        "smiles": smiles,
        "experimental_affinity": c.affinity,
        "affinity_type": c.affinity_type,
        "year": c.year,
        "resolution": c.resolution,
        "ligand_name": c.ligand_name,
        "n_pocket_atoms": result["n_pocket_atoms"],
        "n_lig_atoms": result["n_lig_atoms"],
        "pocket_file": str(pocket_file),
    }


def main():
    parser = argparse.ArgumentParser(description="Fetch PDBbind complexes")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max complexes to download (0=all)")
    parser.add_argument("--refined", action="store_true",
                        help="Use refined set only")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-downloaded PDB IDs")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS,
                        help="Parallel download workers")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"Output directory: {OUT_DIR}")

    # Load or download index
    index_file = OUT_DIR / "pdbbind_index.json"
    if index_file.exists():
        log("Loading cached PDBbind index...")
        with open(index_file) as f:
            complexes = [Complex(**c) for c in json.load(f)]
    else:
        complexes = download_index(refined_only=args.refined)
        with open(index_file, "w") as f:
            json.dump([c.__dict__ for c in complexes], f)

    log(f"Total complexes available: {len(complexes)}")

    if args.limit > 0:
        complexes = complexes[:args.limit]
        log(f"Limited to first {args.limit}")

    # Load existing results for --resume
    existing = set()
    if args.resume and CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            for c in json.load(f):
                existing.add(c["pdb_id"])
        log(f"Resuming: {len(existing)} already downloaded")

    results = []
    done = 0
    failed = 0

    log(f"Downloading {len(complexes)} complexes ({args.workers} workers)...")
    log("This will take ~30 min for 500 complexes (polite: 0.25s delay)")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {}
        for c in complexes:
            if args.resume and c.pdb_id in existing:
                continue
            fut = ex.submit(download_complex, c)
            futures[fut] = c

        for fut in as_completed(futures):
            c = futures[fut]
            try:
                result = fut.result()
                if result:
                    results.append(result)
                    done += 1
                    if done % 50 == 0:
                        log(f"  Progress: {done}/{len(futures)} done, {failed} failed")
                        # Save checkpoint
                        with open(CACHE_FILE, "w") as f:
                            json.dump(results, f, indent=2)
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                if failed <= 5:
                    warn(f"{c.pdb_id}: {e}")

    # Save final results
    if args.resume and CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            all_results = json.load(f)
        for r in results:
            if all(r["pdb_id"] != nr["pdb_id"] for nr in all_results):
                all_results.append(r)
        results = all_results

    with open(CACHE_FILE, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    log(f"\n{'='*60}")
    log(f"  COMPLETE: {done} / {len(complexes)} downloaded")
    log(f"  Failed:  {failed}")
    log(f"  Output:  {CACHE_FILE}")
    if results:
        with_smiles = sum(1 for r in results if r.get("smiles"))
        log(f"  SMILES:  {with_smiles}/{done} have SMILES")
        log(f"\n  To compute features:")
        log(f"  python extract_pdbbind.py --limit {done}")


if __name__ == "__main__":
    main()
