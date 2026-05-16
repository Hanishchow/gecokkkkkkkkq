"""
GEOCK Enhanced Feature Extractor
- Ligand: ECFP4 fingerprints
- Physics: Molecular descriptors (RDKit)
- Protein: Pocket features
"""

import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski, rdMolDescriptors
from collections import Counter

def ecfp4_fingerprint(mol, radius=2, n_bits=512):
    """Morgan fingerprint (ECFP4)"""
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, useChirality=True)
    arr = np.zeros((n_bits,), dtype=np.float32)
    for i in range(n_bits):
        arr[i] = fp.GetBit(i)
    return arr

def physics_features(mol):
    """Molecular physics descriptors"""
    if mol is None:
        return np.zeros(25, dtype=np.float32)
    
    try:
        features = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Lipinski.NumRotatableBonds(mol),
            Lipinski.NumAromaticRings(mol),
            mol.GetNumHeavyAtoms(),
            Descriptors.NumRadicalElectrons(mol),
            Descriptors.NumValenceElectrons(mol),
            Descriptors.FractionCSP3(mol),
            Descriptors.NumAliphaticRings(mol),
            Descriptors.NumSaturatedRings(mol),
            Descriptors.RingCount(mol),
            Descriptors.HeavyAtomMolWt(mol),
            Descriptors.MaxAbsPartialCharge(mol),
            Descriptors.MaxPartialCharge(mol),
            Descriptors.MinAbsPartialCharge(mol),
            Descriptors.MinPartialCharge(mol),
            Descriptors.NumHeteroatoms(mol),
            Descriptors.Flexibility(mol),
            Crippen.MolLogP(mol),
            Lipinski.NumAliphaticHeterocycles(mol),
            Lipinski.NumAliphaticCarbocycles(mol),
            Lipinski.NumSaturatedHeterocycles(mol),
        ]
        return np.array(features, dtype=np.float32)
    except:
        return np.zeros(25, dtype=np.float32)

def protein_pocket_features(pdb_path):
    """
    Extract features from protein pocket PDB file.
    Returns: 50-dimensional feature vector
      - 20 AA composition (normalized)
      - 5 additional amino acid categories (H, E, C, polar, nonpolar)
      - 10 structural features
      - 15 physicochemical properties
    """
    N_AA = 20  # 20 common amino acids
    N_EXTRA = 5  # Additional categories
    N_STRUCT = 10  # Structural features
    N_PHYS = 15  # Physicochemical
    TOTAL = N_AA + N_EXTRA + N_STRUCT + N_PHYS  # = 50
    
    if not Path(pdb_path).exists():
        return np.zeros(TOTAL, dtype=np.float32)
    
    try:
        mol = Chem.MolFromPDBFile(str(pdb_path), removeHs=False)
        if mol is None or mol.GetNumAtoms() == 0:
            return np.zeros(TOTAL, dtype=np.float32)
        
        # Get residue/amino acid composition
        res_types = Counter()
        for atom in mol.GetAtoms():
            res = atom.GetPDBResidueInfo()
            if res and res.GetResidueName():
                res_types[res.GetResidueName().strip()] += 1
        
        # Common amino acids
        common_aa = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 
                    'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
                    'THR', 'TRP', 'TYR', 'VAL']
        
        total = sum(res_types.values()) if res_types else 1
        aa_features = [res_types.get(aa, 0) / total for aa in common_aa]
        
        # Additional categories
        # Hydrophobic: ALA, VAL, LEU, ILE, MET, PHE, TRP
        # Negatively charged: ASP, GLU
        # Positively charged: LYS, ARG, HIS
        # Polar: SER, THR, ASN, GLN, TYR, CYS
        # Other: GLY, PRO
        hydrophobic = sum(res_types.get(aa, 0) for aa in ['ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP'])
        neg_charged = res_types.get('ASP', 0) + res_types.get('GLU', 0)
        pos_charged = res_types.get('LYS', 0) + res_types.get('ARG', 0) + res_types.get('HIS', 0)
        polar = sum(res_types.get(aa, 0) for aa in ['SER', 'THR', 'ASN', 'GLN', 'TYR', 'CYS'])
        other = res_types.get('GLY', 0) + res_types.get('PRO', 0)
        extra_features = [hydrophobic / total, neg_charged / total, pos_charged / total, 
                       polar / total, other / total]
        
        # Structural features
        num_atoms = mol.GetNumAtoms()
        num_heavy = mol.GetNumHeavyAtoms()
        num_carbons = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)
        num_nitrogens = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7)
        num_oxygens = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 8)
        num_sulfurs = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 16)
        num_heteros = num_nitrogens + num_oxygens + num_sulfurs
        
        # Calculate charge balance
        charge_balance = abs(neg_charged - pos_charged) / total if total > 0 else 0
        polar_ratio = polar / total if total > 0 else 0
        
        struct_features = [
            num_atoms / 1000,  # Normalized
            num_heavy / 500,
            num_carbons / max(num_heavy, 1),
            num_nitrogens / max(num_heavy, 1),
            num_oxygens / max(num_heavy, 1),
            num_sulfurs / max(num_heavy, 1),
            num_heteros / max(num_heavy, 1),
            charge_balance,
            polar_ratio,
            hydrophobic / total if total > 0 else 0,
        ]
        
        # Physicochemical (use counts normalized)
        phys_features = [
            hydrophobic / total,  # hydrophobic ratio
            neg_charged / total,
            pos_charged / total,
            polar / total,
            other / total,
            (res_types.get('GLY', 0) + res_types.get('ALA', 0)) / total,  # small aa
            (res_types.get('ILE', 0) + res_types.get('LEU', 0) + res_types.get('VAL', 0)) / total,  # branched
            (res_types.get('PHE', 0) + res_types.get('TYR', 0) + res_types.get('TRP', 0)) / total,  # aromatic
            (res_types.get('CYS', 0)) / total,  # cysteine (for disulfide)
            (res_types.get('MET', 0) + res_types.get('MET', 0)) / total,  # sulfur
            num_sulfurs / max(num_heavy, 1),  # S ratio
            num_heteros / max(num_atoms, 1),  # hetero ratio
            (num_nitrogens + num_oxygens) / max(num_atoms, 1),  # polar atoms ratio
            charge_balance,
            num_atoms / max(num_heavy, 1),  # heavy atom ratio
        ]
        
        # Combine all
        features = aa_features + extra_features + struct_features + phys_features
        
        assert len(features) == TOTAL, f"Expected {TOTAL}, got {len(features)}"
        return np.array(features, dtype=np.float32)
        
    except Exception as e:
        return np.zeros(TOTAL, dtype=np.float32)

def extract_all_features(ligand_path, pocket_path=None):
    """
    Extract all features for a complex.
    
    Returns:
        dict with 'ligand' (512), 'physics' (25), 'pocket' (50)
    """
    features = {}
    
    # Ligand fingerprint
    mol = None
    for ext in ['mol2', 'sdf']:
        p = Path(ligand_path)
        if not p.exists():
            continue
        if ext == 'mol2':
            mol = Chem.MolFromMol2File(str(p), removeHs=False)
        else:
            suppl = Chem.SDMolSupplier(str(p), removeHs=False)
            mol = next(iter(suppl), None)
        if mol:
            break
    
    features['ligand'] = ecfp4_fingerprint(mol)
    features['physics'] = physics_features(mol)
    features['pocket'] = protein_pocket_features(pocket_path) if pocket_path else np.zeros(50, dtype=np.float32)
    
    return features

def combine_features(features_dict, use_ligand=True, use_physics=True, use_pocket=True):
    """Combine features into single vector"""
    parts = []
    if use_ligand and features_dict.get('ligand') is not None:
        parts.append(features_dict['ligand'])
    if use_physics and features_dict.get('physics') is not None:
        parts.append(features_dict['physics'])
    if use_pocket and features_dict.get('pocket') is not None:
        parts.append(features_dict['pocket'])
    
    if not parts:
        return None
    
    return np.concatenate(parts)

if __name__ == "__main__":
    # Test
    CASF = Path("/mnt/c/Users/yakka/Downloads/CASF-2016/CASF-2016/coreset/1a30")
    
    feats = extract_all_features(
        ligand_path=CASF / "1a30_ligand.mol2",
        pocket_path=CASF / "1a30_pocket.pdb"
    )
    
    print("Feature extraction test:")
    print(f"  Ligand (ECFP4): {feats['ligand'].shape}")
    print(f"  Physics: {feats['physics'].shape}")
    print(f"  Pocket: {feats['pocket'].shape}")
    print(f"  Combined: {combine_features(feats).shape}")
    
    # Show some physics features
    print(f"\\nPhysics sample: {feats['physics'][:5]}")