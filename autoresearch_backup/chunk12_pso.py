#!/usr/bin/env python3
"""
Chunk 12: Particle Swarm Optimization (PSO) for Hyperparameter Tuning
"""
import pickle
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
# Import path helpers
try:
    from geock_paths import get_cache_dir, get_work_dir, CACHE_DIR, WORK_DIR
except ImportError:
    from pathlib import Path; CACHE_DIR = Path("/home/chow/.cache/geock_autoresearch"); WORK_DIR = Path("/home/chow/autoresearch")

print("="*70)
print("CHUNK 12: Particle Swarm Optimization (PSO)")
print("="*70)

# Load data
cache = CACHE_DIR / lp_new_features_8k.pkl')
with open(cache, 'rb') as f:
    compounds = pickle.load(f)

with open('CACHE_DIR / physics_features_8k.pkl', 'rb') as f:
    phys_data = pickle.load(f)
X_phys = phys_data['X_phys']

X_int = np.load('WORK_DIR / X_interactions.npy')
with open('WORK_DIR / interaction_pdb_ids.pkl', 'rb') as f:
    int_pdb_ids = pickle.load(f)
int_map = {pdb: i for i, pdb in enumerate(int_pdb_ids)}

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

X_list, y_list = [], []

for i, c in enumerate(compounds):
    ecfp = np.array(c['ecfp'], dtype=np.float32)
    mol = Chem.MolFromSmiles(c['smiles'])
    if mol is None:
        continue
    
    pdb_id = c['pdb_id']
    int_feat = X_int[int_map[pdb_id]] if pdb_id in int_map else np.zeros(20, dtype=np.float32)
    
    mol_feat = np.array([
        Lipinski.RingCount(mol),
        Lipinski.NumAromaticRings(mol),
        Descriptors.MolLogP(mol),
        Descriptors.MolWt(mol),
        ecfp.sum(),
        Lipinski.NumHAcceptors(mol),
        Lipinski.NumHDonors(mol),
        Lipinski.NumRotatableBonds(mol),
    ], dtype=np.float32)
    
    X = np.concatenate([ecfp, mol_feat, X_phys[i], int_feat])
    X_list.append(X)
    y_list.append(c['affinity'])

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.float32)
print(f"Features: {X.shape}")

mu, sd = X.mean(0), X.std(0)
sd[sd == 0] = 1
X_s = (X - mu) / sd

np.random.seed(42)
n = len(X)
perm = np.random.permutation(n)
n_test = int(n * 0.1)
n_val = int(n * 0.1)
n_train = n - n_test - n_val

idx_tr = perm[:n_train]
idx_vl = perm[n_train:n_train+n_val]
idx_te = perm[n_train+n_val:]

X_tr_s = X_s[idx_tr]
X_vl_s = X_s[idx_vl]
X_te_s = X_s[idx_te]
y_tr, y_vl, y_te = y[idx_tr], y[idx_vl], y[idx_te]

print(f"Split: {n_train}/{n_val}/{n_test}")

# PSO Implementation
class ParticleSwarmOptimizer:
    def __init__(self, n_particles=20, n_iterations=30, dim=6):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.dim = dim
        
        # Bounds for each parameter
        # [max_depth, learning_rate, reg_alpha, reg_lambda, n_estimators, min_child_weight]
        self.bounds = np.array([
            [4, 12],        # max_depth
            [0.01, 0.2],    # learning_rate
            [0.1, 2.0],     # reg_alpha
            [1.0, 15.0],    # reg_lambda
            [100, 600],     # n_estimators
            [1, 10],        # min_child_weight
        ])
        
        # Initialize particles
        self.positions = np.random.uniform(
            self.bounds[:, 0], 
            self.bounds[:, 1], 
            (n_particles, dim)
        )
        self.velocities = np.zeros((n_particles, dim))
        
        # Personal best
        self.personal_best_pos = self.positions.copy()
        self.personal_best_vals = np.zeros(n_particles)
        
        # Global best
        self.global_best_pos = np.zeros(dim)
        self.global_best_val = -np.inf
        
        # PSO parameters
        self.w = 0.7    # inertia
        self.c1 = 1.5   # cognitive
        self.c2 = 2.0   # social
        
    def _decode_params(self, position):
        return {
            'max_depth': int(round(position[0])),
            'learning_rate': position[1],
            'reg_alpha': position[2],
            'reg_lambda': position[3],
            'n_estimators': int(round(position[4])),
            'min_child_weight': int(round(position[5])),
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
    
    def _evaluate(self, position):
        params = self._decode_params(position)
        try:
            model = xgb.XGBRegressor(
                random_state=42,
                verbosity=0,
                n_jobs=-1,
                **params
            )
            model.fit(X_tr_s, y_tr)
            pred = model.predict(X_vl_s)
            return pearsonr(y_vl, pred)[0]
        except:
            return -999
    
    def optimize(self):
        print("Running PSO...")
        
        for iteration in range(self.n_iterations):
            # Evaluate all particles
            for i in range(self.n_particles):
                val = self._evaluate(self.positions[i])
                
                # Update personal best
                if val > self.personal_best_vals[i]:
                    self.personal_best_vals[i] = val
                    self.personal_best_pos[i] = self.positions[i].copy()
                
                # Update global best
                if val > self.global_best_val:
                    self.global_best_val = val
                    self.global_best_pos = self.positions[i].copy()
            
            # Update velocities and positions
            r1, r2 = np.random.rand(2)
            
            cognitive = self.c1 * r1 * (self.personal_best_pos - self.positions)
            social = self.c2 * r2 * (self.global_best_pos - self.positions)
            
            self.velocities = self.w * self.velocities + cognitive + social
            
            # Clamp velocities
            vel_range = (self.bounds[:, 1] - self.bounds[:, 0]) * 0.2
            self.velocities = np.clip(self.velocities, -vel_range, vel_range)
            
            self.positions += self.velocities
            
            # Clamp positions to bounds
            self.positions = np.clip(
                self.positions,
                self.bounds[:, 0],
                self.bounds[:, 1]
            )
            
            if (iteration + 1) % 5 == 0:
                print(f"  Iter {iteration+1}: Best R = {self.global_best_val:.4f}")
        
        return self.global_best_pos, self.global_best_val

# Run PSO
pso = ParticleSwarmOptimizer(n_particles=15, n_iterations=25, dim=6)
best_pos, best_r = pso.optimize()

print(f"\nBest PSO params: Val R = {best_r:.4f}")
best_params = pso._decode_params(best_pos)
print(f"Params: {best_params}")

# Train final model with best params
print("\n--- Training final model ---")
seeds = [42, 123, 456, 789, 1000]
all_preds_vl = []
all_preds_te = []

for seed in seeds:
    m = xgb.XGBRegressor(
        random_state=seed,
        verbosity=0,
        n_jobs=-1,
        **best_params
    )
    m.fit(X_tr_s, y_tr)
    all_preds_vl.append(m.predict(X_vl_s))
    all_preds_te.append(m.predict(X_te_s))

ens_vl = np.mean(all_preds_vl, axis=0)
ens_te = np.mean(all_preds_te, axis=0)

r_final_vl = pearsonr(y_vl, ens_vl)[0]
r_final_te = pearsonr(y_te, ens_te)[0]
mae = np.mean(np.abs(y_te - ens_te))

print(f"Ensemble: Val R={r_final_vl:.4f}, Test R={r_final_te:.4f}, MAE={mae:.3f}")

# 5-fold CV
print("\n--- 5-Fold CV ---")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rs = []

for fold, (tr_idx, vl_idx) in enumerate(kf.split(X_s)):
    fold_preds = []
    for seed in seeds:
        m = xgb.XGBRegressor(
            random_state=seed,
            verbosity=0,
            n_jobs=-1,
            **best_params
        )
        m.fit(X_s[tr_idx], y[tr_idx])
        fold_preds.append(m.predict(X_s[vl_idx]))
    
    fold_ens = np.mean(fold_preds, axis=0)
    cv_rs.append(pearsonr(y[vl_idx], fold_ens)[0])
    print(f"  Fold {fold+1}: {cv_rs[-1]:.4f}")

cv_mean = np.mean(cv_rs)
cv_std = np.std(cv_rs)
print(f"\nCV R: {cv_mean:.4f} ± {cv_std:.4f}")

# Save
output = {
    'method': 'PSO',
    'best_params': best_params,
    'cv_r': cv_mean,
    'cv_std': cv_std,
    'val_r': r_final_vl,
    'test_r': r_final_te,
    'mae': mae,
}

with open('WORK_DIR / chunk12_pso_results.pkl', 'wb') as f:
    pickle.dump(output, f)

print(f"\n✓ PSO: CV R={cv_mean:.4f}")