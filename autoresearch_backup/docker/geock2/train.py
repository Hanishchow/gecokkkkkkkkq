"""
train.py - Training script for binding affinity prediction with cross-validation

Supports incremental training: 10 -> 20 -> 30 compounds with 5x5 CV
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from pathlib import Path
import pickle
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from geock.models import (
    PocketGNNAffinity, GNNConfig,
    AttentionVAEAffinity, AttentionVAEConfig,
    atom_features_to_one_hot
)
from geock.pose_utils import pose_to_coordinates, LigandTopology, POSE_DIM
from geock.physics_scoring import PhysicsScorer


@dataclass
class TrainingConfig:
    model_type: str = "gnn"  # "gnn", "vae", or "hybrid"
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    kl_beta: float = 0.1  # For VAE
    patience: int = 15
    min_delta: float = 0.001
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class AffinityDataset(Dataset):
    def __init__(self, samples: List[Dict], max_pocket_atoms: int = 200):
        self.samples = samples
        self.max_pocket_atoms = max_pocket_atoms
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        pose_feat = compute_physics_features(
            sample['ligand_coords'],
            sample['ligand_types'],
            sample['pocket_coords'],
            sample['pocket_features'],
            sample['center']
        )
        
        return {
            'pdb_id': sample['pdb_id'],
            'pocket_features': sample['pocket_features'],
            'pose_features': torch.FloatTensor(pose_feat),
            'physics_features': torch.FloatTensor(pose_feat),  # Use same features
            'affinity': torch.FloatTensor([sample['affinity']])
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function - returns raw pocket features for encoding"""
    return {
        'pdb_id': [b['pdb_id'] for b in batch],
        'pocket_features': [b['pocket_features'] for b in batch],  # Keep as list of numpy arrays
        'pose_features': torch.stack([b['pose_features'] for b in batch]),
        'physics_features': torch.stack([b['physics_features'] for b in batch]),
        'affinity': torch.cat([b['affinity'] for b in batch])
    }


def load_compound_data(compounds_file: str) -> List[Dict]:
    """Load compounds from JSON file"""
    with open(compounds_file) as f:
        return json.load(f)


def parse_pocket_atoms(pocket_pdb: str, center: np.ndarray, radius: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse receptor pocket atoms from PDB file.
    
    Returns:
        coords: [N, 3] atom coordinates
        features: [N, node_features] atom features
    """
    coords = []
    features = []
    
    ATOMIC_MAP = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'S': 16, 'P': 15, 'FE': 26, 'ZN': 30}
    
    with open(pocket_pdb) as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                
                dist = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
                if dist > radius:
                    continue
                
                atom_type = line[76:78].strip()
                if not atom_type:
                    atom_type = line[12:16].strip()[0]
                
                coords.append([x, y, z])
                
                atomic_num = ATOMIC_MAP.get(atom_type.upper(), 6)
                one_hot = np.zeros(100)
                one_hot[atomic_num] = 1.0
                
                is_hydrophobic = 1.0 if atom_type in ['C', 'S'] else 0.0
                is_donor = 1.0 if atom_type in ['N', 'O'] else 0.0
                is_acceptor = 1.0 if atom_type in ['N', 'O', 'S'] else 0.0
                is_aromatic = 0.0
                
                feat = np.concatenate([one_hot, [is_hydrophobic, is_donor, is_acceptor, is_aromatic, dist/radius]])
                features.append(feat)
    
    return np.array(coords), np.array(features)


def parse_ligand_sdf(sdf_path: str) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Parse ligand coordinates from SDF file"""
    coords = []
    atom_types = []
    
    MASSES = {'C': 12.0, 'N': 14.0, 'O': 16.0, 'S': 32.0, 'H': 1.0, 'F': 19.0}
    
    with open(sdf_path) as f:
        lines = f.readlines()
    
    counts = lines[2].split()
    n_atoms = int(counts[0])
    
    for i in range(n_atoms):
        line = lines[3 + i]
        parts = line.split()
        if len(parts) >= 4:
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            atom_type = parts[3] if len(parts) > 3 else 'C'
            coords.append([x, y, z])
            atom_types.append(atom_type)
    
    coords = np.array(coords)
    masses = np.array([MASSES.get(t, 12.0) for t in atom_types])
    
    return coords, atom_types, masses


def prepare_training_data(
    compounds: List[Dict],
    data_dir: str,
    center: np.ndarray,
    radius: float = 10.0,
    max_pocket_atoms: int = 100
) -> List[Dict]:
    """Prepare training samples from compounds"""
    samples = []
    
    for compound in compounds:
        pdb_id = compound['pdb_id']
        exp_affinity = compound['experimental_affinity']
        compound_center = np.array([
            compound['center']['x'],
            compound['center']['y'],
            compound['center']['z']
        ])
        
        pocket_path = os.path.join(data_dir, pdb_id, f"{pdb_id}_pocket.pdb")
        sdf_path = os.path.join(data_dir, pdb_id, f"{pdb_id}_ligand.sdf")
        
        if not os.path.exists(pocket_path) or not os.path.exists(sdf_path):
            print(f"Skipping {pdb_id}: missing files")
            continue
        
        try:
            pocket_coords, pocket_features = parse_pocket_atoms(pocket_path, compound_center, radius)
            ligand_coords, atom_types, masses = parse_ligand_sdf(sdf_path)
            
            if len(pocket_coords) < 5 or len(ligand_coords) < 3:
                print(f"Skipping {pdb_id}: too few atoms")
                continue
            
            # Limit pocket atoms to avoid memory issues
            if len(pocket_coords) > max_pocket_atoms:
                indices = np.random.choice(len(pocket_coords), max_pocket_atoms, replace=False)
                pocket_coords = pocket_coords[indices]
                pocket_features = pocket_features[indices]
            
            samples.append({
                'pdb_id': pdb_id,
                'pocket_coords': pocket_coords,
                'pocket_features': pocket_features,
                'ligand_coords': ligand_coords,
                'ligand_types': atom_types,
                'ligand_masses': masses,
                'center': compound_center,
                'affinity': exp_affinity
            })
        except Exception as e:
            print(f"Error processing {pdb_id}: {e}")
            continue
    
    return samples


def compute_physics_features(
    ligand_coords: np.ndarray,
    ligand_types: List[str],
    pocket_coords: np.ndarray,
    pocket_features: np.ndarray,
    center: np.ndarray
) -> np.ndarray:
    """Compute physics-based features."""
    features = np.zeros(24)
    
    if len(ligand_coords) == 0 or len(pocket_coords) == 0:
        return features
    
    lig_center = ligand_coords.mean(axis=0)
    dist_lig_center = np.linalg.norm(lig_center - center)
    
    all_dists = np.array([np.linalg.norm(lc - pc) for lc in ligand_coords for pc in pocket_coords])
    
    features[0] = np.exp(-dist_lig_center**2 / 8.0)
    features[1] = np.exp(-dist_lig_center**2 / 32.0)
    features[2] = np.exp(-all_dists.min()**2 / 4.0)
    features[3] = np.exp(-all_dists.mean()**2 / 16.0)
    features[4] = min(0, 2.0 - all_dists.min())**2 if all_dists.min() < 2.0 else 0
    
    for i, d in enumerate([2.0, 4.0, 6.0]):
        features[5+i] = np.sum(all_dists < d) / len(all_dists)
    
    features[8] = all_dists.mean()
    features[9] = all_dists.min()
    features[10] = all_dists.std()
    
    n_hydro = sum(1 for t in ligand_types if t in ['C', 'S'])
    n_hbond = sum(1 for t in ligand_types if t in ['N', 'O'])
    features[11] = n_hydro / len(ligand_types)
    features[12] = n_hbond / len(ligand_types)
    features[13] = len(ligand_types) / 50.0
    features[14] = len(pocket_coords) / 100.0
    
    features[15] = np.sin(dist_lig_center / 10.0)
    features[16] = np.cos(dist_lig_center / 10.0)
    
    pocket_types = ['C' if f[6] > 0.5 else ('N' if f[7] > 0.5 else 'O') for f in pocket_features]
    n_rec_hydro = sum(1 for t in pocket_types if t == 'C')
    features[17] = n_rec_hydro / max(1, len(pocket_types))
    
    features[18] = np.exp(-all_dists.mean()**2 / 4.0)
    features[19] = np.exp(-all_dists.mean()**2 / 8.0)
    features[20] = np.exp(-all_dists.mean()**2 / 16.0)
    
    features[21] = np.percentile(all_dists, 25)
    features[22] = np.percentile(all_dists, 50)
    features[23] = np.percentile(all_dists, 75)
    
    return features


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    config: TrainingConfig,
    stats: Optional[Dict] = None
) -> float:
    model.train()
    total_loss = 0
    n_batches = 0
    
    # Get normalization stats
    pose_mean = stats['pose_mean'] if stats else model.pose_mean
    pose_std = stats['pose_std'] if stats else model.pose_std
    phys_mean = stats['phys_mean'] if stats else model.physics_mean
    phys_std = stats['phys_std'] if stats else model.physics_std
    
    for batch in dataloader:
        pose_features = batch['pose_features'].to(config.device)
        physics_features = batch['physics_features'].to(config.device)
        affinities = batch['affinity'].to(config.device)
        
        # Encode each pocket separately and stack embeddings
        pocket_embeddings = []
        for pocket in batch['pocket_features']:
            pocket_tensor = torch.FloatTensor(pocket).to(config.device)
            if pocket_tensor.shape[0] < 200:
                pad = torch.zeros(200 - pocket_tensor.shape[0], pocket_tensor.shape[1])
                pocket_tensor = torch.cat([pocket_tensor, pad], dim=0)
            pocket_emb = model.encode_pocket(pocket_tensor)
            pocket_embeddings.append(pocket_emb)
        
        pocket_embs = torch.stack(pocket_embeddings)
        
        optimizer.zero_grad()
        
        pose_norm = (pose_features - pose_mean) / (pose_std + 1e-8)
        physics_norm = (physics_features - phys_mean) / (phys_std + 1e-8)
        
        affinity_pred = model.predictor(pocket_embs, pose_norm, physics_norm)
        
        # Use MSE loss
        loss = nn.MSELoss()(affinity_pred, affinities.squeeze())
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    config: TrainingConfig,
    stats: Optional[Dict] = None
) -> Tuple[float, float, float]:
    model.eval()
    all_preds = []
    all_targets = []
    
    pose_mean = stats['pose_mean'] if stats else model.pose_mean
    pose_std = stats['pose_std'] if stats else model.pose_std
    phys_mean = stats['phys_mean'] if stats else model.physics_mean
    phys_std = stats['phys_std'] if stats else model.physics_std
    
    with torch.no_grad():
        for batch in dataloader:
            pose_features = batch['pose_features'].to(config.device)
            physics_features = batch['physics_features'].to(config.device)
            
            # Encode each pocket
            pocket_embeddings = []
            for pocket in batch['pocket_features']:
                pocket_tensor = torch.FloatTensor(pocket).to(config.device)
                if pocket_tensor.shape[0] < 200:
                    pad = torch.zeros(200 - pocket_tensor.shape[0], pocket_tensor.shape[1])
                    pocket_tensor = torch.cat([pocket_tensor, pad], dim=0)
                pocket_emb = model.encode_pocket(pocket_tensor)
                pocket_embeddings.append(pocket_emb)
            
            pocket_embs = torch.stack(pocket_embeddings)
            
            pose_norm = (pose_features - pose_mean) / (pose_std + 1e-8)
            physics_norm = (physics_features - phys_mean) / (phys_std + 1e-8)
            
            affinity_pred = model.predictor(pocket_embs, pose_norm, physics_norm)
            
            all_preds.extend(affinity_pred.cpu().numpy())
            all_targets.extend(batch['affinity'].numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    pearson_r, _ = pearsonr(all_preds, all_targets) if len(set(all_targets)) > 1 else (0.0, 1.0)
    mae = mean_absolute_error(all_targets, all_preds)
    mse = np.mean((all_preds - all_targets)**2)
    
    return pearson_r, mae, mse


def cross_validate(
    samples: List[Dict],
    config: TrainingConfig,
    n_folds: int = 5
) -> Dict:
    """5x5 cross-validation training"""
    n_samples = len(samples)
    indices = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    fold_size = n_samples // n_folds
    results = []
    
    for fold in range(n_folds):
        print(f"\n=== Fold {fold + 1}/{n_folds} ===")
        
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size if fold < n_folds - 1 else n_samples
        
        val_indices = indices[val_start:val_end]
        train_indices = np.concatenate([indices[:val_start], indices[val_end:]])
        
        train_samples = [samples[i] for i in train_indices]
        val_samples = [samples[i] for i in val_indices]
        
        train_loader = DataLoader(AffinityDataset(train_samples), batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(AffinityDataset(val_samples), batch_size=config.batch_size, collate_fn=collate_fn)
        
        # Compute stats from training data for normalization
        all_pose = []
        all_phys = []
        for sample in train_samples:
            feat = compute_physics_features(
                sample['ligand_coords'], sample['ligand_types'],
                sample['pocket_coords'], sample['pocket_features'], sample['center']
            )
            all_pose.append(feat)
            all_phys.append(feat)
        
        all_pose = np.array(all_pose)
        all_phys = np.array(all_phys)
        
        stats = {
            'pose_mean': torch.FloatTensor(all_pose.mean(axis=0)).to(config.device),
            'pose_std': torch.FloatTensor(all_pose.std(axis=0) + 1e-8).to(config.device),
            'phys_mean': torch.FloatTensor(all_phys.mean(axis=0)).to(config.device),
            'phys_std': torch.FloatTensor(all_phys.std(axis=0) + 1e-8).to(config.device),
        }
        
        if config.model_type == "vae":
            model_config = AttentionVAEConfig(node_features=105, pose_features=24, physics_features=24)
            model = AttentionVAEAffinity(model_config).to(config.device)
        else:
            model_config = GNNConfig(node_features=105, pose_features=24, physics_features=24)
            model = PocketGNNAffinity(model_config).to(config.device)
        
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=7, factor=0.5, min_lr=1e-5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(config.epochs):
            train_loss = train_epoch(model, train_loader, optimizer, config, stats)
            val_r, val_mae, val_mse = evaluate(model, val_loader, config, stats)
            
            scheduler.step(val_mse)
            
            if epoch % 10 == 0 or epoch == config.epochs - 1:
                print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_r={val_r:.3f}, val_mae={val_mae:.3f}")
            
            if val_mse < best_val_loss - config.min_delta:
                best_val_loss = val_mse
                patience_counter = 0
                best_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= config.patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        model.load_state_dict(best_state)
        final_r, final_mae, _ = evaluate(model, val_loader, config, stats)
        
        results.append({
            'fold': fold + 1,
            'pearson_r': final_r,
            'mae': final_mae,
            'n_train': len(train_samples),
            'n_val': len(val_samples)
        })
        
        print(f"Fold {fold + 1} Results: r={final_r:.3f}, MAE={final_mae:.3f}")
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--compounds', default='/mnt/c/Users/yakka/Downloads/geock_110_data/compounds.json')
    parser.add_argument('--data_dir', default='/mnt/c/Users/yakka/Downloads/geock_110_data')
    parser.add_argument('--output_dir', default='/mnt/c/Users/yakka/Downloads/GEOCK_Project')
    parser.add_argument('--model', default='gnn', choices=['gnn', 'vae', 'hybrid'])
    parser.add_argument('--n_compounds', type=int, default=10)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    print(f"Loading compounds from {args.compounds}")
    compounds = load_compound_data(args.compounds)[:args.n_compounds]
    print(f"Using {len(compounds)} compounds")
    
    print("Preparing training data...")
    samples = prepare_training_data(compounds, args.data_dir, center=np.zeros(3))
    print(f"Prepared {len(samples)} training samples")
    
    config = TrainingConfig(
        model_type=args.model,
        epochs=args.epochs,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"\nStarting {args.folds}x{args.folds} cross-validation...")
    print(f"Model: {args.model}, Device: {config.device}")
    
    results = cross_validate(samples, config, n_folds=args.folds)
    
    r_values = [r['pearson_r'] for r in results]
    mae_values = [r['mae'] for r in results]
    
    print("\n=== Cross-Validation Results ===")
    print(f"Mean Pearson r: {np.mean(r_values):.3f} (+/- {np.std(r_values):.3f})")
    print(f"Mean MAE: {np.mean(mae_values):.3f} (+/- {np.std(mae_values):.3f})")
    
    output_file = os.path.join(args.output_dir, f"cv_results_{args.model}_{args.n_compounds}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(output_file, 'w') as f:
        results_serializable = [{k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                for k, v in r.items()} for r in results]
        json.dump({'config': vars(config), 'results': results_serializable}, f, indent=2)
    
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
