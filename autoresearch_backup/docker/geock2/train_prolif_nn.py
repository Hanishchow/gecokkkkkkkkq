"""
train_prolif_nn.py - Neural Network Training with ProLIF Interaction Fingerprints

Uses ProLIF to compute protein-ligand interaction fingerprints, then trains
a neural network for binding affinity prediction.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
import json
import os
from pathlib import Path

import prolif as plf
from rdkit import Chem

class InteractionFingerprintDataset(Dataset):
    """Dataset using ProLIF interaction fingerprints."""
    
    def __init__(self, X, y, max_len=500):
        self.max_len = max_len
        X_padded = []
        for x in X:
            if len(x) < max_len:
                x_padded = np.pad(x, (0, max_len - len(x)))
            else:
                x_padded = x[:max_len]
            X_padded.append(x_padded)
        self.X = torch.FloatTensor(np.array(X_padded))
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class AffinityMLP(nn.Module):
    """MLP for binding affinity prediction."""
    
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.4):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)


def compute_prolif_fingerprint(ligand_json_path, pocket_pdb_path, n_interactions=9):
    """Compute ProLIF interaction fingerprint for a ligand-protein complex."""
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    import json
    
    try:
        with open(ligand_json_path) as f:
            data = json.load(f)
        
        mol = Chem.RWMol()
        for c in data['coords']:
            try:
                atom = Chem.Atom(c['elem'])
                mol.AddAtom(atom)
            except:
                pass
        
        if mol.GetNumAtoms() == 0:
            return None
        
        conf = Chem.Conformer(len(data['coords']))
        for i, c in enumerate(data['coords']):
            try:
                conf.SetAtomPosition(i, (c['x'], c['y'], c['z']))
            except:
                pass
        mol.AddConformer(conf)
        lmol = plf.Molecule(mol.GetMol())
        
        try:
            pmol_rdkit = Chem.MolFromPDBFile(pocket_pdb_path, sanitize=False, removeHs=False)
        except:
            return None
        
        if pmol_rdkit is None:
            return None
        
        pmol = plf.Molecule(pmol_rdkit)
        
        fp = plf.Fingerprint()
        ifp = fp.generate(lmol, pmol)
        
        all_values = []
        for key, vals in ifp.items():
            all_values.extend(vals)
        arr = np.array(all_values, dtype=np.float32)
        return arr
        
    except Exception as e:
        return None


def load_pdbbind_data(compounds_file, data_dir, max_compounds=None):
    """Load PDBbind data with ProLIF fingerprints."""
    with open(compounds_file, 'r') as f:
        compounds = json.load(f)
    
    if max_compounds:
        compounds = compounds[:max_compounds]
    
    X_list = []
    y_list = []
    ids = []
    errors = []
    
    for i, comp in enumerate(compounds):
        pdb_id = comp['pdb_id']
        affinity = comp['experimental_affinity']
        
        ligand_json = os.path.join(data_dir, pdb_id, f"{pdb_id}_ligand.json")
        pocket_pdb = os.path.join(data_dir, pdb_id, f"{pdb_id}_pocket.pdb")
        
        fp = compute_prolif_fingerprint(ligand_json, pocket_pdb)
        
        if fp is not None and len(fp) > 0:
            X_list.append(fp)
            y_list.append(affinity)
            ids.append(pdb_id)
        else:
            errors.append(pdb_id)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{len(compounds)} compounds, {len(X_list)} valid")
    
    return X_list, np.array(y_list), ids, errors


def train_epoch(model, dataloader, optimizer, criterion):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    for X, y in dataloader:
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    preds = []
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            loss = criterion(pred, y)
            total_loss += loss.item()
            preds.extend(pred.numpy())
    return total_loss / len(dataloader), np.array(preds)


def cross_validate(X, y, n_splits=5, n_epochs=100, lr=0.001, hidden_dims=[256, 128, 64], patience=15):
    """Cross-validation with early stopping."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []
    
    max_len = max(len(x) for x in X)
    print(f"Max fingerprint length: {max_len}")
    
    scaler = StandardScaler()
    X_np = np.array([np.pad(x, (0, max_len - len(x))) if len(x) < max_len else x[:max_len] for x in X])
    X_scaled = scaler.fit_transform(X_np)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n=== Fold {fold + 1}/{n_splits} ===")
        
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        train_dataset = InteractionFingerprintDataset(X_train, y_train, max_len=max_len)
        val_dataset = InteractionFingerprintDataset(X_val, y_val, max_len=max_len)
        
        train_loader = DataLoader(train_dataset, batch_size=min(16, len(train_dataset)), shuffle=True, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=min(16, len(val_dataset)), shuffle=False)
        
        model = AffinityMLP(max_len, hidden_dims=hidden_dims, dropout=0.3)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        best_val_loss = float('inf')
        best_preds = None
        patience_counter = 0
        
        for epoch in range(n_epochs):
            train_loss = train_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_preds = evaluate(model, val_loader, criterion)
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_preds = val_preds.copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 20 == 0:
                r = pearsonr(best_preds, y_val)[0] if len(np.unique(y_val)) > 1 else 0
                print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, r={r:.3f}")
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        r = pearsonr(best_preds, y_val)[0] if len(np.unique(y_val)) > 1 else 0
        mae = mean_absolute_error(y_val, best_preds)
        results.append({
            'fold': fold,
            'pearson_r': r,
            'mae': mae,
            'val_loss': best_val_loss,
            'val_preds': best_preds.tolist(),
            'val_true': y_val.tolist()
        })
        print(f"Fold {fold+1} - Pearson r: {r:.3f}, MAE: {mae:.3f}")
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='/mnt/c/Users/yakka/Downloads/geock_110_data/compounds.json')
    parser.add_argument('--dir', default='/mnt/c/Users/yakka/Downloads/geock_110_data')
    parser.add_argument('--n', type=int, default=30, help='Number of compounds to use')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--hidden', nargs='+', type=int, default=[256, 128, 64])
    parser.add_argument('--output', default='cv_results_prolif_nn.json')
    args = parser.parse_args()
    
    print(f"Loading {args.n} compounds from {args.dir}...")
    X, y, ids, errors = load_pdbbind_data(args.data, args.dir, max_compounds=args.n)
    
    print(f"\nLoaded {len(X)} samples with variable-length ProLIF features")
    print(f"Affinity range: {y.min():.2f} to {y.max():.2f} kcal/mol")
    if errors:
        print(f"Errors: {errors}")
    
    print(f"\nTraining with {args.folds}-fold CV, epochs={args.epochs}, lr={args.lr}")
    results = cross_validate(
        X, y, 
        n_splits=args.folds,
        n_epochs=args.epochs,
        lr=args.lr,
        hidden_dims=args.hidden
    )
    
    all_preds = []
    all_true = []
    for r in results:
        all_preds.extend(r['val_preds'])
        all_true.extend(r['val_true'])
    
    overall_r = pearsonr(all_preds, all_true)[0]
    overall_mae = mean_absolute_error(all_true, all_preds)
    
    print(f"\n{'='*50}")
    print(f"OVERALL RESULTS (ProLIF + NN)")
    print(f"{'='*50}")
    print(f"Pearson r: {overall_r:.3f}")
    print(f"MAE: {overall_mae:.3f} kcal/mol")
    print(f"Target: r > 0.5, MAE < 2.0")
    print(f"Status: {'PASS' if overall_r > 0.5 and overall_mae < 2.0 else 'FAIL'}")
    
    output = {
        'overall_r': float(overall_r),
        'overall_mae': float(overall_mae),
        'n_samples': len(X),
        'fold_results': results
    }
    
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
