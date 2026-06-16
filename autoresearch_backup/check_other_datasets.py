"""Check other datasets we might have"""
import pickle, numpy as np
from pathlib import Path

BACKUP = Path(r'C:\Users\yakka\Downloads\Hanish_Brain_Vault\autoresearch_backup')

# hybrid data
h = pickle.load(open(BACKUP / 'hybrid_compounds.pkl', 'rb'))
print('=== hybrid_compounds.pkl ===')
print(f'Type: {type(h)}, len={len(h)}')
if len(h) > 0:
    if isinstance(h[0], dict):
        print(f'Keys: {list(h[0].keys())[:15]}')
    else:
        print(f'First: {h[0][:3] if isinstance(h[0], (list,np.ndarray)) else h[0]}')

hc = np.load(BACKUP / 'hybrid_features_combined.npy')
print(f'\nhybrid_features_combined.npy: {hc.shape}, {hc.dtype}')

# Also check hybrid features ecfp
he = np.load(BACKUP / 'hybrid_features_ecfp.npy')
print(f'hybrid_features_ecfp.npy: {he.shape}, {he.dtype}')
hy = np.load(BACKUP / 'hybrid_targets.npy')
print(f'hybrid_targets.npy: {hy.shape}, {hy.dtype}')

# X_enhanced
xe = np.load(BACKUP / 'X_enhanced.npy')
print(f'\nX_enhanced.npy: {xe.shape}, {xe.dtype}')

# y.npy
yf = np.load(BACKUP / 'y.npy')
print(f'y.npy: {yf.shape}, {yf.dtype}')

# training_enhanced
te = pickle.load(open(BACKUP / 'geock_training_enhanced.pkl', 'rb'))
print(f'\ngeock_training_enhanced.pkl')
print(f'Type: {type(te)}')
if isinstance(te, dict):
    print(f'Keys: {list(te.keys())[:15]}')
    for k in list(te.keys())[:5]:
        v = te[k]
        if hasattr(v, 'shape'):
            print(f'  {k}: shape={v.shape}')
        elif isinstance(v, list):
            print(f'  {k}: len={len(v)}')
        else:
            print(f'  {k}: {type(v).__name__}')

# X_interactions
xi = np.load(BACKUP / 'X_interactions.npy')
print(f'\nX_interactions.npy: {xi.shape}, {xi.dtype}')
# interaction pdb ids
ip = pickle.load(open(BACKUP / 'interaction_pdb_ids.pkl', 'rb'))
print(f'interaction_pdb_ids.pkl: {type(ip)}, len={len(ip)}')

# test_200
t200 = pickle.load(open(BACKUP / 'test_200_compounds.pkl', 'rb'))
print(f'\ntest_200_compounds.pkl: type={type(t200)}, len={len(t200)}')
if len(t200) > 0:
    if isinstance(t200[0], dict):
        print(f'Keys: {list(t200[0].keys())}')
    else:
        print(f'First: {t200[0][:3]}')
