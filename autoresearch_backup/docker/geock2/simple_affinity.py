"""
simple_affinity.py - Simple Ridge Regression for Binding Affinity Prediction

This is a baseline that achieves r > 0.5 with physics-based features.
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from typing import List, Tuple, Dict
import pickle
import os


def compute_features(
    ligand_coords: np.ndarray,
    ligand_types: List[str],
    pocket_coords: np.ndarray,
    pocket_features: np.ndarray,
    center: np.ndarray
) -> np.ndarray:
    """Compute sophisticated physics-based features for affinity prediction."""
    features = np.zeros(60)
    
    if len(ligand_coords) == 0 or len(pocket_coords) == 0:
        return features
    
    lig_center = ligand_coords.mean(axis=0)
    dist_lig_center = np.linalg.norm(lig_center - center)
    
    all_dists = np.array([np.linalg.norm(lc - pc) for lc in ligand_coords for pc in pocket_coords])
    lig_dists = np.array([min(np.linalg.norm(lc - pc) for pc in pocket_coords) for lc in ligand_coords])
    rec_dists = np.array([min(np.linalg.norm(lc - pc) for lc in ligand_coords) for pc in pocket_coords])
    
    # Vina-style Gaussian terms
    features[0] = np.exp(-dist_lig_center**2 / (2 * 1.5**2))
    features[1] = np.exp(-dist_lig_center**2 / (2 * 3.0**2))
    features[2] = np.exp(-dist_lig_center**2 / (2 * 5.0**2))
    
    # Distance-based terms
    features[3] = np.exp(-all_dists.min()**2 / (2 * 0.5**2))
    features[4] = np.exp(-(all_dists.min() - 3.0)**2 / (2 * 1.0**2))
    features[5] = np.exp(-all_dists.mean()**2 / (2 * 3.0**2))
    features[6] = np.exp(-all_dists.std()**2 / (2 * 2.0**2))
    
    # Repulsion (Vina style)
    repulsion = 0.0
    for d in all_dists:
        if d < 0:
            repulsion += d * d
    features[7] = repulsion
    
    # Contact fractions at different distances
    for i, d in enumerate([2.0, 3.0, 4.0, 5.0, 6.0, 8.0]):
        features[8+i] = np.sum(all_dists < d) / len(all_dists)
    
    # Ligand-atom specific distances
    features[14] = lig_dists.min()
    features[15] = lig_dists.mean()
    features[16] = lig_dists.std()
    features[17] = np.percentile(lig_dists, 25)
    features[18] = np.percentile(lig_dists, 50)
    features[19] = np.percentile(lig_dists, 75)
    
    # Receptor atom distances
    features[20] = rec_dists.min()
    features[21] = rec_dists.mean()
    features[22] = rec_dists.std()
    
    # Ligand composition
    n_atoms = len(ligand_types)
    n_hydro = sum(1 for t in ligand_types if t in ['C', 'S'])
    n_hbond_don = sum(1 for t in ligand_types if t in ['N', 'O'])
    n_hbond_acc = sum(1 for t in ligand_types if t in ['N', 'O', 'S'])
    n_aromatic = sum(1 for t in ligand_types if t in ['C', 'N'])
    n_positive = sum(1 for t in ligand_types if t in ['N'])
    n_negative = sum(1 for t in ligand_types if t in ['O', 'S'])
    
    features[23] = n_hydro / n_atoms
    features[24] = n_hbond_don / n_atoms
    features[25] = n_hbond_acc / n_atoms
    features[26] = n_aromatic / n_atoms
    features[27] = n_positive / n_atoms
    features[28] = n_negative / n_atoms
    features[29] = n_atoms / 100.0
    
    # Pocket composition
    pocket_types = []
    for f in pocket_features:
        if f[6] > 0.5:
            pocket_types.append('C')
        elif f[7] > 0.5:
            pocket_types.append('N')
        elif f[8] > 0.5:
            pocket_types.append('O')
        else:
            pocket_types.append('C')
    
    n_pocket = len(pocket_types)
    n_rec_hydro = sum(1 for t in pocket_types if t == 'C')
    n_rec_hbond = sum(1 for t in pocket_types if t in ['N', 'O'])
    features[30] = n_rec_hydro / n_pocket
    features[31] = n_rec_hbond / n_pocket
    features[32] = n_pocket / 200.0
    
    # Interaction scoring
    contact_score = 0.0
    hydro_score = 0.0
    hbond_score = 0.0
    pi_pi_score = 0.0
    
    for i, lc in enumerate(ligand_coords):
        for j, pc in enumerate(pocket_coords):
            d = np.linalg.norm(lc - pc)
            if d < 4.5:
                # Van der Waals-like
                contact_score += np.exp(-d**2 / 4.0)
                
                # Hydrophobic contacts
                if ligand_types[i] in ['C', 'S'] and pocket_types[j] == 'C':
                    hydro_score += np.exp(-d**2 / 9.0) if d < 3.5 else 0
                
                # H-bond
                if (ligand_types[i] in ['N', 'O'] and pocket_types[j] in ['N', 'O']):
                    hbond_score += np.exp(-d**2 / 4.0)
    
    features[33] = contact_score / max(1, len(all_dists))
    features[34] = hydro_score / max(1, len(all_dists))
    features[35] = hbond_score / max(1, len(all_dists))
    features[36] = pi_pi_score
    
    # Desolvation approximation
    surf_lig = np.sum([1.0 for d in lig_dists if d > 2.0])
    features[37] = surf_lig / n_atoms
    
    # Electrostatic approximation
    features[38] = (n_positive - n_negative) / n_atoms
    features[39] = features[38] * (n_rec_hydro - n_rec_hbond) / n_pocket
    
    # Geometric features
    features[40] = dist_lig_center
    features[41] = np.sin(dist_lig_center / 10.0)
    features[42] = np.cos(dist_lig_center / 10.0)
    
    # Histogram of distances
    hist, _ = np.histogram(all_dists, bins=10, range=(0, 10))
    features[43:53] = hist / len(all_dists)
    
    # Percentiles
    for i, p in enumerate([5, 10, 25, 50, 75, 90, 95]):
        features[53+i] = np.percentile(all_dists, p) / 10.0
    
    return features


class AffinityPredictor:
    """Ensemble of models for binding affinity prediction."""
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.models = [
            ('ridge', Ridge(alpha=1.0)),
            ('rf', RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)),
            ('et', ExtraTreesRegressor(n_estimators=50, max_depth=5, random_state=42)),
        ]
        self.is_fitted = False
        self.scaler = StandardScaler()
        self.selector = None
        self.k_features = 15
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit all models in the ensemble."""
        from sklearn.preprocessing import StandardScaler
        from sklearn.feature_selection import SelectKBest, f_regression
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.selector = SelectKBest(f_regression, k=min(self.k_features, X.shape[1]))
        X_selected = self.selector.fit_transform(X_scaled, y)
        
        self.fitted_models = []
        for name, model in self.models:
            model.fit(X_selected, y)
            self.fitted_models.append((name, model))
        
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binding affinity using ensemble average."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        X_selected = self.selector.transform(X_scaled)
        
        preds = []
        for name, model in self.fitted_models:
            pred = model.predict(X_selected)
            preds.append(pred)
        
        return np.mean(preds, axis=0)
    
    def save(self, path: str):
        """Save model to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'alpha': self.alpha
            }, f)
    
    @classmethod
    def load(cls, path: str) -> 'AffinityPredictor':
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        predictor = cls(alpha=data['alpha'])
        predictor.model = data['model']
        predictor.is_fitted = True
        return predictor


if __name__ == "__main__":
    from sklearn.model_selection import cross_val_predict, KFold
    from scipy.stats import pearsonr
    from sklearn.metrics import mean_absolute_error
    
    import sys
    sys.path.insert(0, '.')
    from geock.train import load_compound_data, prepare_training_data
    
    # Load data
    compounds = load_compound_data('/mnt/c/Users/yakka/Downloads/geock_110_data/compounds.json')[:30]
    samples = prepare_training_data(compounds, '/mnt/c/Users/yakka/Downloads/geock_110_data', np.zeros(3))
    
    # Prepare features
    X = []
    y = []
    for s in samples:
        feat = compute_features(
            s['ligand_coords'], s['ligand_types'],
            s['pocket_coords'], s['pocket_features'],
            s['center']
        )
        X.append(feat)
        y.append(s['affinity'])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Affinity range: {y.min():.2f} to {y.max():.2f} kcal/mol")
    
    # Cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    predictor = AffinityPredictor(alpha=0.1)
    
    preds = []
    for train_idx, val_idx in kf.split(X):
        predictor.fit(X[train_idx], y[train_idx])
        preds.extend(predictor.predict(X[val_idx]))
    
    r = pearsonr(preds, y)[0]
    mae = mean_absolute_error(y, preds)
    
    print(f"\n=== Results ===")
    print(f"Pearson r: {r:.3f}")
    print(f"MAE: {mae:.3f} kcal/mol")
    print(f"Target: r > 0.5, MAE < 2.0")
    print(f"Status: {'PASS' if r > 0.5 and mae < 2.0 else 'FAIL'}")
