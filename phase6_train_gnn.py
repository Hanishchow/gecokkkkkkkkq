#!/usr/bin/env python3
"""
PHASE 6: Graph Neural Network for Binding Affinity Prediction
Uses PyTorch Geometric for protein-ligand graph representations.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GINEConv, GlobalAttention, Set2Set
import numpy as np
import pickle
from pathlib import Path
from scipy.stats import pearsonr
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


# Atom and residue feature dimensions
ATOM_DIM = 19  # C, N, O, S, P, etc.
RESIDUE_DIM = 20  # 20 amino acids


def atom_features(atom):
    """Generate node features for an atom."""
    elem = atom.get('element', 'C')
    features = [0] * ATOM_DIM
    
    # Basic element types
    elem_map = {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'P': 4, 'F': 5, 'CL': 6, 'BR': 7, 'I': 8,
                'MG': 9, 'ZN': 10, 'FE': 11, 'MN': 12, 'CA': 13, 'K': 14, 'NA': 15}
    if elem in elem_map:
        features[elem_map[elem]] = 1
    
    return features


def residue_features(resname):
    """Generate node features for protein residue."""
    features = [0] * RESIDUE_DIM
    
    res_map = {'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4, 'GLN': 5, 'GLU': 6,
               'GLY': 7, 'HIS': 8, 'ILE': 9, 'LEU': 10, 'LYS': 11, 'MET': 12,
               'PHE': 13, 'PRO': 14, 'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19}
    
    if resname in res_map:
        features[res_map[resname]] = 1
    
    return features


def pdb_to_graph(pdb_path, ligand_resnames=None):
    """Convert PDB to graph structure."""
    if ligand_resnames is None:
        ligand_resnames = {'LIG', 'ATP', 'NAD', 'COA', 'FAD', 'SAM', 'GTP', 'GDP', 'UNL'}
    
    # Parse PDB
    atoms = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                elem = line[76:78].strip().upper()
                if not elem:
                    elem = line[12:14].strip().upper()
                if elem in ['', 'H', 'HOH']:
                    continue
                
                resname = line[17:20].strip().upper()
                is_ligand = resname in ligand_resnames
                
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                
                atoms.append({
                    'coords': np.array([x, y, z]),
                    'element': elem,
                    'resname': resname,
                    'is_ligand': is_ligand
                })
    
    if len(atoms) < 3:
        return None
    
    # Compute center
    center = np.mean([a['coords'] for a in atoms], axis=0)
    
    # Center coordinates
    for atom in atoms:
        atom['coords'] = atom['coords'] - center
    
    # Build edges based on distance
    edge_index = []
    edge_attr = []
    cutoff = 5.0  # Angstrom
    
    coords = np.array([a['coords'] for a in atoms])
    
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < cutoff:
                edge_index.append([i, j])
                edge_index.append([j, i])
                
                # Edge features: distance, interaction type
                interaction_type = 0
                if atoms[i]['is_ligand'] != atoms[j]['is_ligand']:
                    interaction_type = 1  # interface
                
                edge_attr.append([dist / cutoff, interaction_type])
                edge_attr.append([dist / cutoff, interaction_type])
    
    if len(edge_index) == 0:
        return None
    
    # Node features
    node_features = []
    for atom in atoms:
        if atom['is_ligand']:
            feat = atom_features(atom)
        else:
            feat = residue_features(atom['resname'])
            # Add ligand indicator
            feat.append(1 if atom['is_ligand'] else 0)
        node_features.append(feat)
    
    # Convert to tensors
    x = torch.FloatTensor(node_features)
    edge_index = torch.LongTensor(edge_index).t().contiguous()
    edge_attr = torch.FloatTensor(edge_attr)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def build_dataset(pdb_dir, compounds, output_path, max_files=None):
    """Build graph dataset."""
    pdb_files = sorted([f for f in os.listdir(pdb_dir) if f.endswith('.pdb')])
    
    if max_files:
        pdb_files = pdb_files[:max_files]
    
    # PDB ID to affinity mapping
    id_to_affinity = {c['pdb_id']: c['affinity'] for c in compounds}
    
    print(f'Processing {len(pdb_files)} PDB files...')
    
    graphs = []
    labels = []
    ids = []
    failed = 0
    
    for i, pdb_file in enumerate(pdb_files):
        if i % 500 == 0:
            print(f'Progress: {i}/{len(pdb_files)}')
        
        pdb_id = pdb_file.replace('.pdb', '')
        
        if pdb_id not in id_to_affinity:
            continue
        
        pdb_path = os.path.join(pdb_dir, pdb_file)
        graph = pdb_to_graph(pdb_path)
        
        if graph is not None and graph.num_nodes > 5:
            graphs.append(graph)
            labels.append(id_to_affinity[pdb_id])
            ids.append(pdb_id)
        else:
            failed += 1
    
    print(f'Done: {len(graphs)} graphs, {failed} failed')
    
    # Save
    output = {
        'graphs': graphs,
        'labels': labels,
        'ids': ids
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(output, f)
    
    print(f'Saved to {output_path}')
    return output


class BindingGNN(nn.Module):
    """GNN for binding affinity prediction."""
    
    def __init__(self, node_dim=ATOM_DIM+1, hidden_dim=128, num_layers=3, dropout=0.3):
        super().__init__()
        
        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GINEConv(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )))
        
        self.set2set = Set2Set(hidden_dim, processing_steps=3)
        
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        
        x = F.relu(self.node_embedding(x))
        
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_attr))
        
        # Pooling
        h = self.set2set(x, data.batch)
        
        # Classification head
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        h = self.dropout(h)
        h = self.fc3(h)
        
        return h.squeeze(-1)


def train_gnn(data_path, output_path, epochs=100, batch_size=32, lr=1e-3):
    """Train GNN model."""
    
    print('Loading data...')
    with open(data_path, 'rb') as f:
        data_dict = pickle.load(f)
    
    graphs = data_dict['graphs']
    labels = torch.FloatTensor(data_dict['labels'])
    ids = data_dict['ids']
    
    print(f'Data: {len(graphs)} graphs')
    
    # Split
    n = len(graphs)
    np.random.seed(42)
    perm = np.random.permutation(n)
    n_test = int(n * 0.1)
    n_val = int(n * 0.1)
    n_train = n - n_test - n_val
    
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train+n_val]
    test_idx = perm[n_train+n_val:]
    
    # Create DataLoader
    train_data = [graphs[i] for i in train_idx]
    val_data = [graphs[i] for i in val_idx]
    test_data = [graphs[i] for i in test_idx]
    
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    test_labels = labels[test_idx]
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    
    # Get node_dim from first graph
    node_dim = train_data[0].x.size(1)
    model = BindingGNN(node_dim=node_dim).to(device)
    
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
        for batch in train_loader:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            pred = model(batch)
            loss = criterion(pred, batch.y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validate
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch)
                val_preds.extend(pred.cpu().detach().numpy())
                val_targets.extend(batch.y.cpu().numpy())
        
        val_r = pearsonr(val_targets, val_preds)[0]
        scheduler.step(train_loss / len(train_loader))
        
        if val_r > best_val_r:
            best_val_r = val_r
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
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
    model.to(device)
    
    # Test evaluation
    model.eval()
    test_preds = []
    test_targets = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch)
            test_preds.extend(pred.cpu().detach().numpy())
            test_targets.extend(batch.y.cpu().numpy())
    
    test_r = pearsonr(test_targets, test_preds)[0]
    test_mae = np.mean(np.abs(np.array(test_preds) - np.array(test_targets)))
    
    print(f'\\nTest R: {test_r:.4f}')
    print(f'Test MAE: {test_mae:.3f}')
    
    # Save
    torch.save({
        'model_state_dict': best_model_state,
        'val_r': best_val_r,
        'test_r': test_r,
        'test_mae': test_mae
    }, output_path)
    
    print(f'Model saved to {output_path}')
    
    return model, test_r


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train GNN for binding affinity')
    parser.add_argument('--pdb-dir', default='CACHE_DIR / lp_pdb_files')
    parser.add_argument('--labels', default='CACHE_DIR / lp_new_features_8k.pkl')
    parser.add_argument('--graphs', default='WORK_DIR / graphs.pkl')
    parser.add_argument('--output', default='WORK_DIR / geock_gnn.pt')
    parser.add_argument('--max', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--build-only', action='store_true', help='Only build graphs, skip training')
    
    args = parser.parse_args()
    
    # First build graphs if needed
    if not Path(args.graphs).exists():
        print('Building graph dataset...')
        with open(args.labels, 'rb') as f:
            compounds = pickle.load(f)
        build_dataset(args.pdb_dir, compounds, args.graphs, max_files=args.max)
    
    if not args.build_only:
        train_gnn(args.graphs, args.output, epochs=args.epochs, 
                  batch_size=args.batch_size, lr=args.lr)