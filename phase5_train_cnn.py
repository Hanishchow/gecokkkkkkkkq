#!/usr/bin/env python3
"""
PHASE 5: 3D CNN Model for Binding Affinity Prediction
Uses 3D voxel grids of protein-ligand complexes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from pathlib import Path
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')


class BindingDataset(Dataset):
    """Dataset for 3D binding affinity prediction."""
    
    def __init__(self, grids, y, ids):
        self.grids = torch.FloatTensor(grids)
        self.y = torch.FloatTensor(y)
        self.ids = ids
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        # Input: (channels, depth, height, width)
        return self.grids[idx], self.y[idx]


class Binding3DCNN(nn.Module):
    """3D CNN for binding affinity prediction."""
    
    def __init__(self, in_channels=40, hidden_dims=[32, 64, 128], dropout=0.3):
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_channels, hidden_dims[0], kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(hidden_dims[0])
        
        self.conv2 = nn.Conv3d(hidden_dims[0], hidden_dims[1], kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(hidden_dims[1])
        
        self.conv3 = nn.Conv3d(hidden_dims[1], hidden_dims[2], kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(hidden_dims[2])
        
        self.pool = nn.MaxPool3d(2)
        self.dropout = nn.Dropout3d(dropout)
        
        # Calculate feature size after convolutions
        # Input: 24 -> 12 -> 6 -> 3
        self.fc1 = nn.Linear(hidden_dims[2] * 3 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, x):
        # Conv block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        # Conv block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        # Conv block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x.squeeze(-1)


def train_3d_cnn(grids_path, labels_path, output_path, epochs=100, batch_size=16, lr=1e-4):
    """Train 3D CNN model."""
    
    # Load data
    print('Loading data...')
    with open(grids_path, 'rb') as f:
        grid_data = pickle.load(f)
    
    grids = grid_data['grids']
    ids = grid_data['ids']
    
    # Load labels - match by PDB ID
    with open(labels_path, 'rb') as f:
        compounds = pickle.load(f)
    
    # Create ID to affinity mapping
    id_to_affinity = {c['pdb_id']: c['affinity'] for c in compounds}
    
    # Filter grids with labels
    valid_grids = []
    valid_y = []
    valid_ids = []
    
    for i, pdb_id in enumerate(ids):
        if pdb_id in id_to_affinity:
            valid_grids.append(grids[i])
            valid_y.append(id_to_affinity[pdb_id])
            valid_ids.append(pdb_id)
    
    grids = np.array(valid_grids)
    y = np.array(valid_y)
    
    print(f'Data: {len(y)} samples, grid shape {grids.shape}')
    
    # Split
    n = len(y)
    np.random.seed(42)
    perm = np.random.permutation(n)
    n_test = int(n * 0.1)
    n_val = int(n * 0.1)
    n_train = n - n_test - n_val
    
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train+n_val]
    test_idx = perm[n_train+n_val:]
    
    train_ds = BindingDataset(grids[train_idx], y[train_idx], [valid_ids[i] for i in train_idx])
    val_ds = BindingDataset(grids[val_idx], y[val_idx], [valid_ids[i] for i in val_idx])
    test_ds = BindingDataset(grids[test_idx], y[test_idx], [valid_ids[i] for i in test_idx])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    
    model = Binding3DCNN().to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()
    
    # Training
    best_val_r = 0
    best_model_state = None
    patience_counter = 0
    patience = 20
    
    print('Training...')
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for X, y_batch in train_loader:
            X, y_batch = X.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validate
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for X, y_batch in val_loader:
                X = X.to(device)
                pred = model(X)
                val_preds.extend(pred.cpu().numpy())
                val_targets.extend(y_batch.numpy())
        
        val_r = pearsonr(val_targets, val_preds)[0]
        scheduler.step(train_loss / len(train_loader))
        
        if val_r > best_val_r:
            best_val_r = val_r
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Train loss={train_loss/len(train_loader):.4f}, Val R={val_r:.4f}')
        
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Test evaluation
    model.eval()
    test_preds = []
    test_targets = []
    with torch.no_grad():
        for X, y_batch in test_loader:
            X = X.to(device)
            pred = model(X)
            test_preds.extend(pred.cpu().numpy())
            test_targets.extend(y_batch.numpy())
    
    test_r = pearsonr(test_targets, test_preds)[0]
    test_mae = np.mean(np.abs(np.array(test_preds) - np.array(test_targets)))
    
    print(f'\\nTest R: {test_r:.4f}')
    print(f'Test MAE: {test_mae:.3f}')
    
    # Save model
    torch.save({
        'model_state_dict': best_model_state,
        'val_r': best_val_r,
        'test_r': test_r,
        'test_mae': test_mae,
        'config': {
            'in_channels': 40,
            'hidden_dims': [32, 64, 128],
            'dropout': 0.3,
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr
        }
    }, output_path)
    
    print(f'Model saved to {output_path}')
    
    return model, test_r


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train 3D CNN for binding affinity')
    parser.add_argument('--grids', default='WORK_DIR / 3d_grids.pkl')
    parser.add_argument('--labels', default='CACHE_DIR / lp_new_features_8k.pkl')
    parser.add_argument('--output', default='WORK_DIR / geock_cnn_3d.pt')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    
    args = parser.parse_args()
    
    train_3d_cnn(args.grids, args.labels, args.output, 
                  epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)