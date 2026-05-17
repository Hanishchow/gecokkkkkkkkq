#!/usr/bin/env python3
"""
extract_fast.py — Fast feature extraction for 100+ compounds
Skips VQE and IntegrationFilter for speed. Uses SMILES from compounds.json.
"""
import os, sys, json, pickle, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/chow/autoresearch")

import numpy as np
from pathlib import Path

CACHE_DIR  = Path("/home/chow/.cache/geock_autoresearch")
DATA_DIR   = Path("/mnt/c/Users/yakka/Downloads/geock_110_data")
OUT_CACHE  = CACHE_DIR / "features_110.pkl"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_NAMES_24 = [
    "E1_vinardo_gauss1", "E1_vinardo_repulsion", "E1_vinardo_hydrophobic",
    "E1_vinardo_hbond", "E1_vinardo_torsion", "E1_vinardo_affinity",
    "E2_chem_pi_pi", "E2_chem_cation_pi", "E2_chem_salt_bridge",
    "E2_chem_halogen_bond", "E2_chem_metal_coord", "E2_chem_burial",
    "E2_chem_shape", "E2_chem_lipophilic",
    "E3_quantum_vqe",
    "E4_bio_drug_likeness", "E4_bio_ligand_efficiency",
    "E4_bio_pocket_druggability", "E4_bio_resolution_weight",
    "E4_bio_family_hydrophobic", "E4_bio_family_hbond",
    "E4_bio_pocket_polarity", "E4_bio_size_penalty", "E4_bio_pharmacophore",
]

def get_ecfp(smiles, fp_size=512):
    try:
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(fp_size, dtype=np.float32)
        fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fp_size)
        return np.array(fpgen.GetFingerprintAsNumPy(mol), dtype=np.float32)
    except:
        return np.zeros(fp_size, dtype=np.float32)

def drug_likeness_from_smiles(smiles):
    """Compute Lipinski + Veber drug-likeness score from SMILES."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0
        
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        rotb = Descriptors.NumRotatableBonds(mol)
        tpsa = Descriptors.TPSA(mol)
        
        score = 0.0
        if mw <= 500: score += 0.2
        if logp <= 5: score += 0.2
        if hbd <= 5: score += 0.2
        if hba <= 10: score += 0.2
        if rotb <= 10: score += 0.1
        if tpsa <= 140: score += 0.1
        return score
    except:
        return 0.0

def ligand_efficiency(smiles, affinity_kcal):
    """LE = affinity / n_heavy_atoms."""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0
        n_heavy = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
        if n_heavy == 0:
            return 0.0
        return abs(affinity_kcal) / n_heavy
    except:
        return 0.0

def pocket_residue_features(rec_types):
    """Simple pocket composition features."""
    na = rec_types.count("NA")
    oa = rec_types.count("OA")
    sa = rec_types.count("SA")
    c  = rec_types.count("C")
    total = len(rec_types)
    if total == 0:
        return 0.0, 0.0, 0.0
    
    polar = (na + oa + sa) / total
    hydrophobic = c / total
    buried = 1.0 - polar * 0.5  # rough buriedness score
    return polar, hydrophobic, buried

def biological_features_simple(rec_types, smiles, affinity_kcal, pdb_id=""):
    """
    Fast biological features (no VQE, no RCSB lookup).
    Uses SMILES + pocket composition for all E4 features.
    """
    feats = np.zeros(9, dtype=np.float32)  # E4: indices 15-23
    
    # drug_likeness (idx 15) — Lipinski + Veber
    feats[0] = drug_likeness_from_smiles(smiles)
    
    # ligand_efficiency (idx 16) — |affinity| / n_heavy_atoms
    feats[1] = ligand_efficiency(smiles, affinity_kcal)
    
    # Pocket composition
    na = rec_types.count("NA")
    oa = rec_types.count("OA")
    sa = rec_types.count("SA")
    c  = rec_types.count("C")
    n  = len(rec_types)
    if n == 0:
        return feats
    
    polar_ratio = (na + oa + sa) / n
    c_ratio     = c / n
    
    # pocket_druggability (idx 17) — hydrophobic buried pocket
    feats[2] = min(1.0, (c_ratio * 0.7 + (1 - polar_ratio) * 0.3) * 1.2)
    
    # resolution_weight (idx 18) — based on n_residue (proxy for resolution)
    # More residues → larger pocket → possibly lower-res structure
    feats[3] = np.clip(1.0 - (n - 20) / 200.0, 0.3, 1.0)
    
    # family_hydrophobic (idx 19) — fraction of C atoms
    feats[4] = c_ratio
    
    # family_hbond (idx 20) — polar atom ratio
    feats[5] = polar_ratio
    
    # pocket_polarity (idx 21) — same as polar_ratio
    feats[6] = polar_ratio
    
    # size_penalty (idx 22) — penalize extreme pocket sizes
    # Optimal: 20-100 residues. Too small or too large = penalty
    feats[7] = np.clip(1.0 - abs(n - 50) / 100.0, 0.0, 1.0)
    
    # pharmacophore (idx 23) — SA count as sulfur pharmacophore
    feats[8] = min(1.0, sa / max(1, n))
    
    return feats

def compute_physics_and_chem(rec, rt, lig, lt, n_torsions=0):
    """
    Compute E1 (Vinardo) + E2 (chemistry) + E3 (quantum=0) features.
    Skips VQE for speed.
    """
    feats = np.zeros(24, dtype=np.float32)
    
    try:
        from enhanced_physics import vinardo_features, chemistry_features
        e1 = vinardo_features(rec, rt, lig, lt, n_torsions=n_torsions)
        e2 = chemistry_features(rec, rt, lig, lt)
        feats[:6] = e1[:6]
        feats[6:14] = e2[:8]
        feats[14] = 0.0  # E3 quantum = 0 (skipped)
    except Exception as e:
        pass
    
    return feats

def process_compound(c):
    """Process one compound, return 24D features + affinity."""
    pdb_id = c["pdb_id"]
    smiles = c.get("smiles", "")
    aff_kcal = c["experimental_affinity"]
    pdb_path = DATA_DIR / pdb_id / f"{pdb_id}_pocket.pdb"
    
    try:
        from patch_parse import parse_pocket_and_ligand
        rec, rt, lig, lt, _, nrot = parse_pocket_and_ligand(str(pdb_path))
    except:
        return None
    
    # E1+E2+E3 (physics)
    feats = compute_physics_and_chem(rec, rt, lig, lt, n_torsions=nrot)
    
    # E4 (biological) — with SMILES
    e4 = biological_features_simple(rt, smiles, aff_kcal, pdb_id)
    feats[15:24] = e4
    
    return feats.astype(np.float32), aff_kcal

def main():
    print("=" * 65)
    print("  FAST FEATURE EXTRACTION")
    print("=" * 65)
    
    with open(DATA_DIR / "compounds.json") as f:
        comps = json.load(f)
    
    valid = []
    for c in comps:
        pdb_path = DATA_DIR / c["pdb_id"] / f"{c['pdb_id']}_pocket.pdb"
        smi = c.get("smiles", "")
        if not pdb_path.exists() or not smi:
            continue
        from rdkit import Chem
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
        except:
            continue
        valid.append(c)
    
    print(f"  Valid compounds: {len(valid)}")
    
    # Check cache
    if OUT_CACHE.exists():
        with open(OUT_CACHE, "rb") as f:
            old = pickle.load(f)
        old_ids = set(old["pdb_ids"])
        missing_ids = [c["pdb_id"] for c in valid if c["pdb_id"] not in old_ids]
        print(f"  Already cached: {len(old_ids)}")
        print(f"  Missing: {len(missing_ids)}")
        use_cache = (len(missing_ids) == 0)
    else:
        old = None
        use_cache = False
    
    if use_cache and old is not None:
        print(f"  Using existing cache.")
        X_raw_all = old["X_raw"]
        y_all = old["y_pkd"]
        ids_all = old["pdb_ids"]
        ecfp_all = old["X_ecfp"]
        feat_data = old
    else:
        print(f"\n  Processing {len(valid)} compounds...")
        start = time.time()
        
        X_list, y_list, ids_list = [], [], []
        pdb_to_smi = {c["pdb_id"]: c.get("smiles", "") for c in valid}
        
        for i, c in enumerate(valid):
            result = process_compound(c)
            if result is None:
                print(f"    FAILED {c['pdb_id']}")
                continue
            
            feats, aff = result
            X_list.append(feats)
            y_list.append(aff)
            ids_list.append(c["pdb_id"])
            
            if (i+1) % 25 == 0:
                elapsed = time.time() - start
                rate = (i+1) / elapsed
                remaining = (len(valid) - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1:3d}/{len(valid)}] {c['pdb_id']}: "
                      f"vina={feats[5]:+.1f} drug={feats[15]:.2f} "
                      f"lig_eff={feats[16]:.3f} eta={remaining:.0f}s")
        
        elapsed = time.time() - start
        print(f"\n  Processed {len(X_list)} compounds in {elapsed:.1f}s")
        
        X_raw_all = np.array(X_list, dtype=np.float32)
        y_raw = np.array(y_list, dtype=np.float32)
        y_all = (-y_raw / 1.364).astype(np.float32)
        
        # ECFP4
        print("\n  Computing ECFP4...")
        ecfp_list = [get_ecfp(pdb_to_smi.get(pid, "")) for pid in ids_list]
        ecfp_all = np.array(ecfp_list, dtype=object)
        
        feat_data = {
            "X_raw": X_raw_all,
            "y_pkd": y_all,
            "pdb_ids": ids_list,
            "X_ecfp": ecfp_all,
            "n_compounds": len(ids_list),
        }
        with open(OUT_CACHE, "wb") as f:
            pickle.dump(feat_data, f)
        print(f"  Cached to {OUT_CACHE}")
    
    # Feature stats
    print("\n" + "=" * 65)
    print("  FEATURE STATS")
    print("=" * 65)
    Xr = feat_data["X_raw"]
    nc = feat_data["n_compounds"]
    for i, n in enumerate(FEATURE_NAMES_24):
        nz = (Xr[:, i] != 0).sum()
        mn = Xr[:, i].mean()
        sd = Xr[:, i].std()
        bar = "#" * (nz * 30 // nc) if nc > 0 else ""
        print(f"  [{i:2d}] {n:30s} {bar:30s} mean={mn:+.3f} std={sd:.3f}")
    
    print(f"\n  Total: {feat_data['n_compounds']} compounds, X_raw={X_raw_all.shape}")
    print(f"  y range: {feat_data['y_pkd'].min():.2f} – {feat_data['y_pkd'].max():.2f} pKd")
    
    # Check drug_likeness
    dl = feat_data["X_raw"][:, 15]
    print(f"\n  drug_likeness: mean={dl.mean():.3f}, non-zero={np.count_nonzero(dl)}/{feat_data['n_compounds']}")
    print(f"  ligand_efficiency: mean={feat_data['X_raw'][:,16].mean():.3f}")
    
    return feat_data

if __name__ == "__main__":
    main()
