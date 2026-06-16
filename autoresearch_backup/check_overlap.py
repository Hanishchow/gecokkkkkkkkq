"""Check data overlap between sources"""
import pickle, numpy as np
from pathlib import Path
import csv

BACKUP = Path(r'C:\Users\yakka\Downloads\Hanish_Brain_Vault\autoresearch_backup')

# 39K merged
m39 = pickle.load(open(BACKUP / 'merged_39k.pkl', 'rb'))
X39 = np.array([e['ecfp'] for e in m39], dtype=np.float32)
y39 = np.array([e['affinity'] for e in m39], dtype=np.float32)
h39 = set(hash(row.tobytes()) for row in X39)
print(f'39K merged: {len(X39)} entries, {len(h39)} unique ECFP')

# Phase 2 19K
X19 = np.load(BACKUP / 'phase2_X.npy')
y19 = np.load(BACKUP / 'phase2_y.npy')
h19 = set(hash(row.tobytes()) for row in X19[:, :512])
print(f'Phase 2 19K: {len(X19)} entries, {len(h19)} unique ECFP')

# Overlap
both = h39 & h19
only39 = h39 - h19
only19 = h19 - h39
print(f'\nOverlap: {len(both)} ECFP vectors')
print(f'Only in 39K: {len(only39)}')
print(f'Only in Phase 2: {len(only19)}')
print(f'Combined unique ECFP: {len(h39 | h19)}')

# Check lp_features_enhanced
try:
    lpe = pickle.load(open(BACKUP / 'lp_features_enhanced.pkl', 'rb'))
    print(f'\nlp_features_enhanced: {len(lpe)} entries')
    if len(lpe) > 0 and isinstance(lpe[0], dict):
        print(f'Keys: {list(lpe[0].keys())}')
        print(f'Features len: {len(lpe[0]["features"])}')
except Exception as e:
    print(f'\nlp_features_enhanced error: {e}')

# Check data.tsv
try:
    with open(BACKUP / 'data.tsv') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        print(f'\ndata.tsv header: {header}')
        rows = list(reader)
        print(f'Rows: {len(rows)}')
        if rows:
            print(f'Sample: {rows[0]}')
except Exception as e:
    print(f'\ndata.tsv error: {e}')

# Count total unique ECFP across BOTH datasets, and find which ones have full 982-dim features
# Build ECFP -> best_entry mapping
X19_ecfp = X19[:, :512]
ecfp_to_19 = {}
for i, h in enumerate([hash(row.tobytes()) for row in X19_ecfp]):
    ecfp_to_19[h] = i

ecfp_to_39 = {}
for i, h in enumerate([hash(row.tobytes()) for row in X39]):
    ecfp_to_39[h] = i

combined = h39 | h19
print(f'\nCombined unique molecules: {len(combined)}')
print(f'  Have full 982-dim features: {len(h19)} ({len(h19)/len(combined)*100:.1f}%)')
print(f'  Have ECFP-only: {len(h39)} ({len(h39)/len(combined)*100:.1f}%)')
print(f'  ECFP-only only (no 982-dim): {len(only39)} ({len(only39)/len(combined)*100:.1f}%)')
