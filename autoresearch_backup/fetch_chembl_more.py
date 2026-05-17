#!/usr/bin/env python3
"""Fetch more ChEMBL data and combine with LP-PDBBind for training."""
import requests
import time
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

CACHE_DIR = '/home/chow/.cache/geock_autoresearch'
N_BITS = 300

def get_chembl_ecfp(smiles, n_bits=N_BITS):
    """Compute ECFP from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits))

def fetch_chembl_page(page, page_size=100):
    """Fetch a page of ChEMBL activity data."""
    base_url = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
    params = {
        'standard_type': 'IC50,Ki,Kd',
        'standard_relation': '=',
        'standard_units': 'nM',
        'limit': page_size,
        'offset': page * page_size + 1,
        'molecule_chembl_id__isnull': False,
        'target_chembl_id__isnull': False
    }
    try:
        r = requests.get(base_url, params=params, timeout=30)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception as e:
        print(f"Error fetching page {page}: {e}")
        return None

def fetch_chembl_continue(n_pages=50, start_page=13):
    """Continue fetching ChEMBL data from where we left off."""
    existing = pickle.load(open(f'{CACHE_DIR}/chembl_tiny.pkl', 'rb'))
    existing_smiles = {c['smiles'] for c in existing}
    print(f"Existing: {len(existing)} compounds, {len(existing_smiles)} unique SMILES")

    new_compounds = []
    seen_smiles = set(existing_smiles)

    for page in range(start_page, start_page + n_pages):
        data = fetch_chembl_page(page)
        if data is None or 'activities' not in data:
            time.sleep(0.5)
            continue

        activities = data.get('activities', [])
        if not activities:
            break

        for act in activities:
            smiles = act.get('molecule_structures', {}).get('canonical_smiles')
            if not smiles:
                continue
            if smiles in seen_smiles:
                continue

            try:
                val = float(act.get('standard_value', 0))
                if val <= 0:
                    continue
                pkd = -np.log10(val * 1e-9) if val > 0 else 0
                if pkd < 0 or pkd > 15:
                    continue
                ecfp = get_chembl_ecfp(smiles)
                if ecfp is None:
                    continue

                new_compounds.append({
                    'ecfp': ecfp,
                    'pKd': pkd,
                    'smiles': smiles
                })
                seen_smiles.add(smiles)
            except:
                continue

        print(f"Page {page}: {len(activities)} activities, {len(new_compounds)} new compounds")
        time.sleep(0.3)

    print(f"Total new: {len(new_compounds)}")
    return existing + new_compounds

if __name__ == '__main__':
    combined = fetch_chembl_continue(n_pages=50, start_page=13)
    print(f"Total combined: {len(combined)}")
    pickle.dump(combined, open(f'{CACHE_DIR}/chembl_more.pkl', 'wb'))
