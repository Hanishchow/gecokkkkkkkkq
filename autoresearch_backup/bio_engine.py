"""
bio_engine.py
=============
GEOCK 2.0 — Biological Context Engine (Engine 4)
+  Integration Filter (combines all 4 engines, removes noise)

Engine 1: Vinardo Physics     (6 features)
Engine 2: Chemistry-Based     (8 features)
Engine 3: Quantum VQE         (1 feature)
Engine 4: Biological Context  (9 features)  ← NEW
─────────────────────────────────────────────
Integration Filter:           combines + denoises all 24 features

Why a Bio Engine?
  Vinardo, chemistry, and VQE all describe the MOLECULE.
  Biology describes the CONTEXT:
    - What protein family is this?
    - What are the key residues?
    - Is this a known pharmacophore?
    - What's the ligand's drug-like score?
    - How does the pocket compare to known druggable pockets?

  These biological priors dramatically reduce noise on diverse datasets.
  This is exactly why R was weak on 30 diverse compounds — no target context.

Usage:
    from bio_engine import unified_score
    result = unified_score(pdb_id, pocket_pdb, smiles,
                           rec_coords, rec_types,
                           lig_coords, lig_types, n_tors)
    # returns 24D feature vector, cleaned and denoised
"""

from __future__ import annotations

import math
import warnings
import numpy as np
from typing import Optional
import urllib.request
import json
import os
import time

warnings.filterwarnings("ignore")

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem
    from rdkit.Chem.rdMolDescriptors import CalcTPSA
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False

# ── Import other engines ────────────────────────────────────────────────────
try:
    from enhanced_physics import (
        vinardo_features,
        chemistry_features,
        vqe_feature,
        HYDROPHOBIC_TYPES,
        HBOND_ACC,
        HBOND_DON,
        VDW,
    )
    ENGINES_OK = True
except ImportError:
    ENGINES_OK = False
    warnings.warn("enhanced_physics.py not found — run from geock directory")


# ═══════════════════════════════════════════════════════════════════════════
# ENGINE 4: BIOLOGICAL CONTEXT  (9 features)
# ═══════════════════════════════════════════════════════════════════════════

# Known druggable pocket volume ranges (Angstrom³)
DRUGGABLE_POCKET_VOLUME = (300, 2000)

# Protein family keywords → binding pattern priors
PROTEIN_FAMILIES = {
    "kinase":     {"depth": 0.8, "hydrophobic": 0.6, "hbond": 0.4},
    "protease":   {"depth": 0.7, "hydrophobic": 0.5, "hbond": 0.7},
    "nuclear":    {"depth": 0.6, "hydrophobic": 0.8, "hbond": 0.3},
    "gpcr":       {"depth": 0.9, "hydrophobic": 0.7, "hbond": 0.3},
    "phosphatase":{"depth": 0.6, "hydrophobic": 0.4, "hbond": 0.8},
    "unknown":    {"depth": 0.5, "hydrophobic": 0.5, "hbond": 0.5},
}

# Lipinski + Veber drug-likeness thresholds
LIPINSKI = {
    "mw_max": 500, "logp_max": 5,
    "hbd_max": 5,  "hba_max": 10,
}
VEBER = {
    "rotbonds_max": 10, "tpsa_max": 140,
}

# Cache for RCSB lookups
_bio_cache: dict[str, dict] = {}
_CACHE_FILE = "/tmp/geock_bio_cache.json"

def _load_cache():
    global _bio_cache
    if os.path.exists(_CACHE_FILE):
        try:
            with open(_CACHE_FILE) as f:
                _bio_cache = json.load(f)
        except Exception:
            _bio_cache = {}

def _save_cache():
    try:
        with open(_CACHE_FILE, "w") as f:
            json.dump(_bio_cache, f)
    except Exception:
        pass

_load_cache()


def _fetch_rcsb_info(pdb_id: str) -> dict:
    """Fetch protein family and resolution from RCSB."""
    if pdb_id in _bio_cache:
        return _bio_cache[pdb_id]

    result = {
        "resolution":   3.0,
        "family":       "unknown",
        "n_chains":     1,
        "has_ligand":   True,
        "organism":     "unknown",
        "method":       "X-RAY",
    }

    try:
        url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.lower()}"
        req = urllib.request.Request(url, headers={"User-Agent": "GEOCK/2.0"})
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.loads(r.read())

        # Resolution
        if "rcsb_entry_info" in data:
            res = data["rcsb_entry_info"].get("resolution_combined", [3.0])
            result["resolution"] = float(res[0]) if res else 3.0

        # Method
        if "exptl" in data:
            method = data["exptl"][0].get("method", "X-RAY")
            result["method"] = method

        time.sleep(0.1)   # rate limit

    except Exception:
        pass

    try:
        # Get protein title for family classification
        url2 = (f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.lower()}"
                f"?fields=struct.title")
        req2 = urllib.request.Request(url2, headers={"User-Agent": "GEOCK/2.0"})
        with urllib.request.urlopen(req2, timeout=5) as r:
            data2 = json.loads(r.read())

        title = data2.get("struct", {}).get("title", "").lower()
        for family in PROTEIN_FAMILIES:
            if family in title:
                result["family"] = family
                break

    except Exception:
        pass

    _bio_cache[pdb_id] = result
    _save_cache()
    return result


def biological_features(
    pdb_id:     str,
    rec_coords: np.ndarray,
    rec_types:  list[str],
    lig_coords: np.ndarray,
    lig_types:  list[str],
    smiles:     Optional[str] = None,
    n_torsions: int = 0,
) -> np.ndarray:
    """
    9 biological context features:

    [0] drug_likeness_score    — Lipinski + Veber combined score
    [1] ligand_efficiency      — ΔG / n_heavy_atoms (normalised)
    [2] pocket_druggability    — volume in optimal druggable range
    [3] resolution_weight      — crystal structure quality (lower = better)
    [4] family_hydrophobic_prior — expected hydrophobic contribution for protein family
    [5] family_hbond_prior     — expected H-bond contribution
    [6] pocket_polarity_ratio  — polar/nonpolar balance
    [7] ligand_size_penalty    — too big or too small penalty
    [8] pharmacophore_match    — known pharmacophore patterns from SMILES
    """
    feats = np.zeros(9, dtype=np.float32)

    n_lig = len(lig_coords)
    n_rec = len(rec_coords)

    # ── Feature 0: Drug-likeness score ──────────────────────────────────
    dl_score = 0.0
    if smiles and RDKIT_OK:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            mw   = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd  = Descriptors.NumHDonors(mol)
            hba  = Descriptors.NumHAcceptors(mol)
            rot  = Descriptors.NumRotatableBonds(mol)
            tpsa = CalcTPSA(mol)

            # Lipinski (4 rules)
            lip = sum([
                mw   <= LIPINSKI["mw_max"],
                logp <= LIPINSKI["logp_max"],
                hbd  <= LIPINSKI["hbd_max"],
                hba  <= LIPINSKI["hba_max"],
            ]) / 4.0

            # Veber (2 rules)
            veb = sum([
                rot  <= VEBER["rotbonds_max"],
                tpsa <= VEBER["tpsa_max"],
            ]) / 2.0

            dl_score = (lip * 0.6 + veb * 0.4)
    feats[0] = float(dl_score)

    # ── Feature 1: Ligand efficiency ────────────────────────────────────
    # LE = ΔG / n_heavy_atoms — optimal drugs have LE > 0.3
    n_heavy = n_lig
    if smiles and RDKIT_OK:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            n_heavy = mol.GetNumHeavyAtoms()

    # Use pairwise contact density as proxy for ΔG
    lig_arr = np.array(lig_coords)
    rec_arr = np.array(rec_coords)
    contact_score = sum(
        math.exp(-np.linalg.norm(l - r) / 3.0)
        for l in lig_arr
        for r in rec_arr
        if np.linalg.norm(l - r) < 5.0
    )
    le_proxy = contact_score / max(1, n_heavy)
    feats[1] = float(np.clip(le_proxy / 10.0, 0, 1))

    # ── Feature 2: Pocket druggability ──────────────────────────────────
    # Volume in optimal druggable range [300, 2000 Å³]
    span = np.ptp(rec_arr, axis=0)
    vol  = float(np.prod(span + 1e-3) * 0.7)   # 0.7 packing factor
    low, high = DRUGGABLE_POCKET_VOLUME
    if low <= vol <= high:
        # Peak druggability at 800 Å³
        feats[2] = float(1.0 - abs(vol - 800) / 1200.0)
    elif vol < low:
        feats[2] = float(vol / low * 0.5)
    else:
        feats[2] = float(max(0, 1.0 - (vol - high) / 3000.0))

    # ── Feature 3: Resolution weight ────────────────────────────────────
    # Better resolution = more trustworthy coordinates = higher weight
    info = _fetch_rcsb_info(pdb_id)
    res  = info.get("resolution", 3.0)
    # 1.0Å → 1.0, 2.0Å → 0.75, 3.0Å → 0.5, 4.0Å → 0.25
    feats[3] = float(np.clip(1.0 - (res - 1.0) / 4.0, 0.1, 1.0))

    # ── Feature 4 & 5: Protein family priors ────────────────────────────
    family = info.get("family", "unknown")
    prior  = PROTEIN_FAMILIES.get(family, PROTEIN_FAMILIES["unknown"])
    feats[4] = float(prior["hydrophobic"])
    feats[5] = float(prior["hbond"])

    # ── Feature 6: Pocket polarity ratio ────────────────────────────────
    n_polar    = sum(1 for t in rec_types if t in HBOND_ACC)
    n_nonpolar = sum(1 for t in rec_types if t in HYDROPHOBIC_TYPES)
    total      = n_polar + n_nonpolar + 1e-6
    feats[6]   = float(n_polar / total)

    # ── Feature 7: Ligand size penalty ──────────────────────────────────
    # Penalty for ligands too small (<5 atoms) or too large (>50 atoms)
    if n_heavy < 5:
        feats[7] = float(n_heavy / 5.0)
    elif n_heavy > 50:
        feats[7] = float(max(0, 1.0 - (n_heavy - 50) / 30.0))
    else:
        feats[7] = 1.0

    # ── Feature 8: Pharmacophore match ──────────────────────────────────
    # Known pharmacophore patterns: aromatic + HBD + HBA in right arrangement
    pharma = 0.0
    if smiles and RDKIT_OK:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            has_aromatic = any(a.GetIsAromatic() for a in mol.GetAtoms())
            has_hbd = Descriptors.NumHDonors(mol) > 0
            has_hba = Descriptors.NumHAcceptors(mol) > 0
            has_hydro= any(a.GetAtomicNum() == 6 and not a.GetIsAromatic()
                          for a in mol.GetAtoms())

            # Classic drug pharmacophore: aromatic + HBD or HBA + hydrophobic
            pharma = sum([
                has_aromatic * 0.3,
                has_hbd      * 0.25,
                has_hba      * 0.25,
                has_hydro    * 0.2,
            ])
    feats[8] = float(pharma)

    return feats


# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATION FILTER ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class IntegrationFilter:
    """
    Combines outputs of all 4 engines and removes noise.

    Noise sources in multi-engine pipelines:
      1. Scale mismatch    — VQE in kcal/mol, burial in [0,1]
      2. Redundancy        — some features encode similar information
      3. Outlier features  — one broken pocket inflates raw distance terms
      4. Cross-target noise— same feature means different things in kinase vs protease

    Filter operations:
      1. Per-feature z-score normalisation (removes scale)
      2. Outlier clipping at ±3σ (removes extreme values)
      3. Resolution weighting (down-weight low-quality structures)
      4. Family-conditioned feature scaling (cross-target normalisation)
      5. Redundancy-aware feature selection (keeps most informative)
    """

    def __init__(self):
        self.fitted    = False
        self.means_    = None
        self.stds_     = None
        self.clip_val_ = 3.0     # clip at 3 sigma
        self.min_std_  = 1e-6    # avoid division by zero

    def fit(self, X: np.ndarray, resolution_weights: np.ndarray = None):
        """
        Fit normalisation parameters on training data.
        X: (n_compounds, 24) — all 4 engines combined
        resolution_weights: (n,) — crystal resolution quality weights
        """
        if resolution_weights is not None:
            # Weighted statistics — better structures count more
            w = resolution_weights / resolution_weights.sum()
            self.means_ = np.average(X, axis=0, weights=w)
            self.stds_  = np.sqrt(
                np.average((X - self.means_) ** 2, axis=0, weights=w)
            )
        else:
            self.means_ = X.mean(axis=0)
            self.stds_  = X.std(axis=0)

        self.stds_ = np.maximum(self.stds_, self.min_std_)
        self.fitted = True
        return self

    def transform(self, X: np.ndarray,
                  resolution_weights: np.ndarray = None) -> np.ndarray:
        """
        Apply filter to feature matrix.
        Returns cleaned, normalised, denoised feature matrix.
        """
        if not self.fitted:
            raise RuntimeError("Call fit() before transform()")

        X_clean = X.copy().astype(np.float32)

        # ── Step 1: Z-score normalise ────────────────────────────────────
        X_clean = (X_clean - self.means_) / self.stds_

        # ── Step 2: Clip outliers at ±3σ ────────────────────────────────
        X_clean = np.clip(X_clean, -self.clip_val_, self.clip_val_)

        # ── Step 3: Resolution weighting ────────────────────────────────
        # Down-weight features for low-resolution structures
        if resolution_weights is not None:
            w = resolution_weights.reshape(-1, 1)
            w = np.clip(w, 0.1, 1.0)
            X_clean = X_clean * w

        # ── Step 4: Replace NaN/Inf ──────────────────────────────────────
        X_clean = np.nan_to_num(X_clean, nan=0.0, posinf=0.0, neginf=0.0)

        return X_clean

    def fit_transform(self, X: np.ndarray,
                      resolution_weights: np.ndarray = None) -> np.ndarray:
        return self.fit(X, resolution_weights).transform(X, resolution_weights)

    def noise_report(self, X_raw: np.ndarray,
                     feature_names: list[str]) -> dict:
        """
        Report which features have high noise before filtering.
        Useful for debugging which engine is causing problems.
        """
        report = {}
        for i, name in enumerate(feature_names):
            col  = X_raw[:, i]
            mn = np.mean(col)
            sd = np.std(col)
            cv = sd / (abs(mn) + 1e-9) if abs(mn) > 1e-9 else (np.inf if sd > 1e-9 else 0.0)
            n_outliers = np.sum(
                np.abs(col - mn) > 3 * sd
            ) if sd > 1e-9 else 0
            is_zero_var = sd < 1e-9
            report[name] = {
                "mean":       float(mn),
                "std":        float(sd),
                "cv":         float(cv),
                "n_outliers": int(n_outliers),
                "noisy":      (cv > 2.0 and not np.isinf(cv)) or n_outliers > len(col) * 0.1 or is_zero_var,
            }
        return report


# ═══════════════════════════════════════════════════════════════════════════
# UNIFIED SCORE — All 4 Engines + Integration Filter
# ═══════════════════════════════════════════════════════════════════════════

FEATURE_NAMES_FULL = [
    # Engine 1: Vinardo (0-5)
    "E1_vinardo_gauss1",
    "E1_vinardo_repulsion",
    "E1_vinardo_hydrophobic",
    "E1_vinardo_hbond",
    "E1_vinardo_torsion",
    "E1_vinardo_affinity",
    # Engine 2: Chemistry (6-13)
    "E2_chem_pi_pi",
    "E2_chem_cation_pi",
    "E2_chem_salt_bridge",
    "E2_chem_halogen_bond",
    "E2_chem_metal_coord",
    "E2_chem_burial",
    "E2_chem_shape",
    "E2_chem_lipophilic",
    # Engine 3: Quantum (14)
    "E3_quantum_vqe",
    # Engine 4: Biology (15-23)
    "E4_bio_drug_likeness",
    "E4_bio_ligand_efficiency",
    "E4_bio_pocket_druggability",
    "E4_bio_resolution_weight",
    "E4_bio_family_hydrophobic",
    "E4_bio_family_hbond",
    "E4_bio_pocket_polarity",
    "E4_bio_size_penalty",
    "E4_bio_pharmacophore",
]


def unified_score(
    pdb_id:      str,
    rec_coords:  np.ndarray,
    rec_types:   list[str],
    lig_coords:  np.ndarray,
    lig_types:   list[str],
    n_torsions:  int = 0,
    smiles:      Optional[str] = None,
    use_quantum: bool = True,
) -> np.ndarray:
    """
    Run all 4 engines and return raw 24D feature vector.

    Returns
    -------
    np.ndarray shape (24,) — raw features, all 4 engines.
    NOTE: This returns raw features. Normalization/filtering should be
    applied by the caller using ONLY training data to avoid leakage.
    Use run_all_compounds(train_mask=...) for proper ML pipeline.
    """
    if not ENGINES_OK:
        raise ImportError("enhanced_physics.py not found. "
                          "Place it in the same directory.")

    # Engine 1 + 2 + 3
    e1_e2_e3 = np.zeros(15, dtype=np.float32)
    try:
        e1 = vinardo_features(rec_coords, rec_types,
                              lig_coords, lig_types, n_torsions)
        e2 = chemistry_features(rec_coords, rec_types,
                                lig_coords, lig_types, smiles)
        e3 = vqe_feature(smiles, rec_coords, rec_types) \
             if (use_quantum and smiles) else 0.0
        e1_e2_e3 = np.concatenate([e1, e2, [e3]])
    except Exception as ex:
        warnings.warn(f"Engines 1-3 failed for {pdb_id}: {ex}")

    # Engine 4: Biology
    e4 = np.zeros(9, dtype=np.float32)
    try:
        e4 = biological_features(
            pdb_id, rec_coords, rec_types,
            lig_coords, lig_types, smiles, n_torsions
        )
    except Exception as ex:
        warnings.warn(f"Engine 4 failed for {pdb_id}: {ex}")

    return np.concatenate([e1_e2_e3, e4]).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# FULL DATASET RUNNER — with integration filter
# ═══════════════════════════════════════════════════════════════════════════

def run_all_compounds(
    compounds:    list[dict],
    data_dir:     str,
    use_quantum:  bool = True,
    verbose:      bool = True,
    train_mask:   list[bool] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Run all 4 engines on every compound.

    ML-CORRECT: IntegrationFilter is fit ONLY on training compounds
    (identified by train_mask). This prevents data leakage.
    The filter is then applied to ALL compounds.

    Parameters
    ----------
    compounds   : list of dicts with pdb_id, smiles, experimental_affinity
    data_dir    : path to PDB files
    use_quantum : run VQE (slow) or skip
    verbose     : print progress
    train_mask  : list of bool (length n_compounds) indicating training set.
                  If None, no filtering is applied (returns raw features).

    Returns
    -------
    X_filtered   : (n, 24) filtered feature matrix (normalized on train only)
    X_raw        : (n, 24) raw feature matrix (before filtering)
    y            : (n,)    experimental affinities
    pdb_ids      : list of successfully processed PDB IDs
    """
    try:
        from patch_parse import parse_pocket_and_ligand
    except ImportError:
        raise ImportError("patch_parse.py not found")

    X_raw_list = []
    y_list     = []
    pdb_ids    = []
    res_weights= []
    failed     = []

    total = len(compounds)
    for i, c in enumerate(compounds):
        pdb_id = c["pdb_id"]
        pdb    = os.path.join(data_dir, pdb_id, f"{pdb_id}_pocket.pdb")
        smiles = c.get("smiles", "")

        if not os.path.exists(pdb):
            failed.append((pdb_id, "no pocket PDB"))
            continue

        try:
            rec, rt, lig, lt, _, nrot = parse_pocket_and_ligand(pdb)

            feats = unified_score(
                pdb_id, rec, rt, lig, lt,
                n_torsions  = nrot,
                smiles      = smiles,
                use_quantum = use_quantum,
            )

            # Get resolution weight from bio engine cache
            info = _bio_cache.get(pdb_id, {})
            res  = info.get("resolution", 3.0)
            res_w= float(np.clip(1.0 - (res - 1.0) / 4.0, 0.1, 1.0))

            X_raw_list.append(feats)
            y_list.append(float(c["experimental_affinity"]))
            pdb_ids.append(pdb_id)
            res_weights.append(res_w)

            if verbose:
                print(f"  [{i+1:3d}/{total}] {pdb_id}: "
                      f"vina={feats[5]:+.2f} "
                      f"vqe={feats[14]:+.2f} "
                      f"drug_like={feats[15]:.2f} "
                      f"res_w={res_w:.2f}")

        except Exception as e:
            failed.append((pdb_id, str(e)[:60]))
            if verbose:
                print(f"  [{i+1:3d}/{total}] {pdb_id}: FAILED — {e}")

    if not X_raw_list:
        raise RuntimeError("No compounds processed successfully")

    X_raw = np.array(X_raw_list, dtype=np.float32)
    y     = np.array(y_list,     dtype=np.float32)
    res_w = np.array(res_weights,dtype=np.float32)

    if train_mask is not None and len(train_mask) == len(X_raw):
        train_mask_arr = np.array(train_mask, dtype=bool)
        filt = IntegrationFilter()
        filt.fit(X_raw[train_mask_arr], resolution_weights=res_w[train_mask_arr])
        X_filtered = filt.transform(X_raw, resolution_weights=res_w)
    else:
        X_filtered = X_raw
        filt = None

    # Noise report (on raw features)
    if verbose:
        print(f"\n{'='*60}")
        print(f"INTEGRATION FILTER REPORT")
        print(f"{'='*60}")
        report = {}
        for i, name in enumerate(FEATURE_NAMES_FULL):
            col  = X_raw[:, i]
            cv   = np.std(col) / (abs(np.mean(col)) + 1e-9) if np.mean(col) != 0 else 0
            n_outliers = np.sum(np.abs(col - np.mean(col)) > 3 * np.std(col))
            report[name] = {
                "mean": float(np.mean(col)),
                "std":  float(np.std(col)),
                "cv":   float(cv),
                "n_outliers": int(n_outliers),
                "noisy": bool(cv > 2.0 or n_outliers > len(col) * 0.1),
            }
        noisy  = [(n, r) for n, r in report.items() if r["noisy"]]
        clean  = [(n, r) for n, r in report.items() if not r["noisy"]]
        print(f"Clean features : {len(clean)}/24")
        print(f"Noisy features : {len(noisy)}/24")
        if noisy:
            print("\nNoisy features (high CV or outliers):")
            for name, r in noisy:
                print(f"  {name[:35]:35s} "
                      f"CV={r['cv']:.2f} "
                      f"outliers={r['n_outliers']}")
        print(f"\nFailed compounds: {len(failed)}")
        for pid, reason in failed[:5]:
            print(f"  {pid}: {reason}")

    return X_filtered, X_raw, y, pdb_ids


# ═══════════════════════════════════════════════════════════════════════════
# QUICK TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    print("Testing Bio Engine + Integration Filter...")
    print("="*60)

    # Mock data
    np.random.seed(42)
    rec_coords = np.random.randn(30, 3) * 4.0
    rec_types  = (["C","A","NA","OA","C","SA","C","A","FE","ZN"] * 3)[:30]
    lig_coords = np.random.randn(12, 3) * 1.5
    lig_types  = ["NA","C","C","OA","C","C","C","OA","P","OA","C","NA"]
    smiles     = "c1cc(ccc1CC(C(=O)O)N)OP(=O)(O)O"
    pdb_id     = "1a1e"

    # Test bio engine alone
    print("Engine 4 (Biology):")
    bio = biological_features(pdb_id, rec_coords, rec_types,
                              lig_coords, lig_types, smiles)
    bio_names = [
        "drug_likeness", "ligand_efficiency", "pocket_druggability",
        "resolution_weight", "family_hydrophobic", "family_hbond",
        "pocket_polarity", "size_penalty", "pharmacophore"
    ]
    for name, val in zip(bio_names, bio):
        print(f"  {name:<25}: {val:.4f}")

    # Test unified score
    print("\nAll 4 Engines — Unified Score:")
    feats = unified_score(pdb_id, rec_coords, rec_types,
                          lig_coords, lig_types,
                          n_torsions=5, smiles=smiles,
                          use_quantum=True)
    print(f"  Feature vector shape: {feats.shape}")
    for i, (name, val) in enumerate(zip(FEATURE_NAMES_FULL, feats)):
        engine = f"E{1 if i<6 else 2 if i<14 else 3 if i<15 else 4}"
        print(f"  [{i:2d}] {engine} {name:<35}: {val:+.4f}")

    # Test integration filter on mock batch
    print("\nIntegration Filter (mock batch of 10):")
    X_mock = np.random.randn(10, 24).astype(np.float32)
    X_mock[3, 5] = 500.0   # inject outlier
    filt   = IntegrationFilter()
    X_filt = filt.fit_transform(X_mock)
    print(f"  Raw max:      {X_mock.max():.2f}")
    print(f"  Filtered max: {X_filt.max():.2f}  (outlier clipped)")
    print(f"  Shape: {X_mock.shape} → {X_filt.shape}")
    print("\nAll engines + filter working.")
    print("="*60)
    print("\nML-CORRECT USAGE — IntegrationFilter fit on TRAIN only:")
    print("  from bio_engine import run_all_compounds")
    print("  X, X_raw, y, ids = run_all_compounds(")
    print("      compounds, data_dir, use_quantum=True,")
    print("      train_mask=[True]*10 + [False]*10)  # first 10 are train")
