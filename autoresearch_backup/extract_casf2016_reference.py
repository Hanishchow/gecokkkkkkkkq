"""Extract CASF-2016 SMILES + pKd reference for Colab evaluation"""
import numpy as np
from pathlib import Path
from rdkit import Chem

CASF_DIR = Path(r'C:\Users\yakka\Downloads\CASF-2016\CASF-2016')
CORESET_DAT = CASF_DIR / 'power_scoring' / 'CoreSet.dat'
OUT = Path(r'C:\Users\yakka\Downloads\Hanish_Brain_Vault\autoresearch_backup\casf2016_reference.csv')

complexes = []
with open(CORESET_DAT, 'r') as f:
    f.readline()
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 4:
            complexes.append({'pdb_id': parts[0], 'pkd': float(parts[3])})

print(f"Found {len(complexes)} CASF-2016 complexes")

with open(OUT, 'w') as f:
    f.write('pdb_id,pkd_true,smiles\n')
    success = 0
    for c in complexes:
        pid = c['pdb_id']
        mol2_path = CASF_DIR / 'coreset' / pid / f'{pid}_ligand.mol2'
        sdf_path = CASF_DIR / 'coreset' / pid / f'{pid}_ligand.sdf'
        mol = None
        if mol2_path.exists():
            mol = Chem.MolFromMol2File(str(mol2_path))
        if mol is None and sdf_path.exists():
            sup = Chem.SDMolSupplier(str(sdf_path))
            if sup: mol = sup[0]
        if mol is not None:
            smiles = Chem.MolToSmiles(mol)
            f.write(f"{pid},{c['pkd']},{smiles}\n")
            success += 1
        else:
            f.write(f"{pid},{c['pkd']},\n")
            print(f"  WARN: No ligand for {pid}")

print(f"Saved {success}/{len(complexes)} complexes to {OUT}")
