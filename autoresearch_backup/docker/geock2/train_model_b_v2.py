"""MODEL B: ElasticNet + ChemBERTa (optimized with batch processing)"""
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
import json, os, sys, time, pickle
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '.')
from patch_parse import parse_pocket_and_ligand

# Load ChemBERTa with shorter sequence
print("Loading ChemBERTa...")
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
chemberta = AutoModel.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
chemberta.eval()

def get_chemberta_embedding_batch(smiles_list, batch_size=16):
    """Batch encode SMILES"""
    embeddings = []
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i+batch_size]
        # Filter empty
        batch = [s if s else "C" for s in batch]
        try:
            inputs = tokenizer(batch, return_tensors='pt', max_length=64, 
                            truncation=True, padding=True)
            with torch.no_grad():
                outputs = chemberta(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1).numpy()
            embeddings.extend(emb)
        except:
            embeddings.extend([np.zeros(768)] * len(batch))
    return np.array(embeddings)

def compute_physics(lig_coords, lig_types, rec_coords, rec_types, center):
    features = np.zeros(60)
    if len(lig_coords) < 3 or len(rec_coords) < 10:
        return features
    
    lig_center = lig_coords.mean(axis=0)
    dist = np.linalg.norm(lig_center - center)
    
    all_dists = np.array([np.linalg.norm(lc - pc) for lc in lig_coords for pc in rec_coords])
    lig_dists = np.array([min(np.linalg.norm(lc - pc) for pc in rec_coords) for lc in lig_coords])
    rec_dists = np.array([min(np.linalg.norm(lc - pc) for lc in lig_coords) for pc in rec_coords])
    
    features[0] = np.exp(-dist**2 / (2 * 1.5**2))
    features[1] = np.exp(-dist**2 / (2 * 3.0**2))
    features[2] = np.exp(-dist**2 / (2 * 5.0**2))
    features[3] = np.exp(-all_dists.min()**2 / (2 * 0.5**2))
    features[4] = np.exp(-(all_dists.min() - 3.0)**2 / (2 * 1.0**2))
    features[5] = np.exp(-all_dists.mean()**2 / (2 * 3.0**2))
    features[6] = np.exp(-all_dists.std()**2 / (2 * 2.0**2))
    features[7] = sum(d * d for d in all_dists if d < 0)
    
    for i, d in enumerate([2.0, 3.0, 4.0, 5.0, 6.0, 8.0]):
        features[8+i] = np.sum(all_dists < d) / len(all_dists)
    
    features[14] = lig_dists.min()
    features[15] = lig_dists.mean()
    features[16] = lig_dists.std()
    features[17] = np.percentile(lig_dists, 25)
    features[18] = np.percentile(lig_dists, 50)
    features[19] = np.percentile(lig_dists, 75)
    features[20] = rec_dists.min()
    features[21] = rec_dists.mean()
    features[22] = rec_dists.std()
    
    n = len(lig_types)
    features[23] = sum(1 for t in lig_types if t in ['C','S']) / n
    features[24] = sum(1 for t in lig_types if t in ['N','O']) / n
    features[25] = sum(1 for t in lig_types if t in ['N','O','S']) / n
    features[26] = sum(1 for t in lig_types if t in ['C','N']) / n
    features[27] = sum(1 for t in lig_types if t == 'N') / n
    features[28] = sum(1 for t in lig_types if t in ['O','S']) / n
    features[29] = n / 100.0
    
    np_ = len(rec_types)
    features[30] = sum(1 for t in rec_types if t == 'C') / np_
    features[31] = sum(1 for t in rec_types if t in ['N','O']) / np_
    features[32] = np_ / 200.0
    
    contact = hydro = hbond = 0.0
    for i, lc in enumerate(lig_coords):
        for j, pc in enumerate(rec_coords):
            d = np.linalg.norm(lc - pc)
            if d < 4.5:
                contact += np.exp(-d**2 / 4.0)
                if lig_types[i] in ['C','S'] and rec_types[j] == 'C':
                    hydro += np.exp(-d**2 / 9.0) if d < 3.5 else 0
                if lig_types[i] in ['N','O'] and rec_types[j] in ['N','O']:
                    hbond += np.exp(-d**2 / 4.0)
    
    features[33] = contact / max(1, len(all_dists))
    features[34] = hydro / max(1, len(all_dists))
    features[35] = hbond / max(1, len(all_dists))
    features[37] = sum(1.0 for d in lig_dists if d > 2.0) / n
    features[38] = (features[27] - features[28]) * n / n
    features[39] = features[38] * (features[30] - features[31])
    features[40] = dist
    features[41] = np.sin(dist / 10.0)
    features[42] = np.cos(dist / 10.0)
    
    hist, _ = np.histogram(all_dists, bins=10, range=(0, 10))
    features[43:53] = hist / len(all_dists)
    for i, p in enumerate([5, 10, 25, 50, 75, 90, 95]):
        features[53+i] = np.percentile(all_dists, p) / 10.0
    
    return features

# Load data
data_dir = '/mnt/c/Users/yakka/Downloads/geock_110_data'
with open(f'{data_dir}/compounds.json') as f:
    compounds = json.load(f)

X_phys, smiles_list, y = [], [], []
t0 = time.time()

print(f"Extracting physics features...")
for i, c in enumerate(compounds):
    pdb_file = f'{data_dir}/{c["pdb_id"]}/{c["pdb_id"]}_pocket.pdb'
    try:
        rec_coords, rec_types, lig_coords, lig_types, _, _ = parse_pocket_and_ligand(pdb_file, cutoff=10.0)
        center = rec_coords.mean(axis=0)
        X_phys.append(compute_physics(lig_coords, lig_types, rec_coords, rec_types, center))
        smiles_list.append(c.get('smiles', ''))
        y.append(c['experimental_affinity'])
    except:
        pass

X_phys = np.array(X_phys)
y = np.array(y)
print(f"Physics features: {len(X_phys)} compounds in {time.time()-t0:.1f}s")

# Get ChemBERTa embeddings in batch
print("Computing ChemBERTa embeddings...")
t1 = time.time()
X_emb = get_chemberta_embedding_batch(smiles_list, batch_size=32)
print(f"ChemBERTa: {X_emb.shape} in {time.time()-t1:.1f}s")

# Combine
X_hybrid = np.hstack([X_phys, X_emb])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_hybrid)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

preds = []
for train_idx, val_idx in kf.split(X_scaled):
    model = ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000)
    model.fit(X_scaled[train_idx], y[train_idx])
    preds.extend(model.predict(X_scaled[val_idx]))

r_b = pearsonr(preds, y)[0]
mae_b = mean_absolute_error(y, preds)

print(f"\n{'='*60}")
print(f"MODEL B RESULTS (ElasticNet + ChemBERTa)")
print(f"{'='*60}")
print(f"Pearson R: {r_b:.4f}")
print(f"MAE: {mae_b:.4f} kcal/mol")

# Compare
r_a = 0.1366
print(f"\n{'='*60}")
print(f"MODEL A (ECFP4): r = {r_a:.4f}")
print(f"MODEL B (ChemBERTa): r = {r_b:.4f}")
print(f"Improvement: {r_b - r_a:+.4f}")

if r_b > r_a + 0.03:
    print(f"\n>>> MODEL B WINS")
    with open('/mnt/c/Users/yakka/Downloads/final/geock/model_b.pkl', 'wb') as f:
        pickle.dump({'model_type': 'ElasticNet', 'alpha': 0.001, 'l1_ratio': 0.5,
                    'scaler': scaler, 'use_chemberta': True}, f)
else:
    print(f"\n>>> MODEL A WINS (keep ECFP4)")
