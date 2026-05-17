import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
with open('CACHE_DIR / lp_all_features.pkl', 'rb') as f:
    data_list = pickle.load(f)

print(f"Total compounds: {len(data_list)}")

# Atom feature extraction
def one_hot_encoding(x, allowable_set):
    return [1 if x == s else 0 for s in allowable_set]

def get_atom_features(atom):
    atomic_num = atom.GetAtomicNum()
    features = one_hot_encoding(atomic_num, [1, 6, 7, 8, 9, 15, 16, 17, 35, 53])  # H, C, N, O, F, P, S, Cl, Br, I
    features += one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
    features += one_hot_encoding(atom.GetFormalCharge(), [-1, 0, 1])
    features += one_hot_encoding(atom.GetHybridization(), [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ])
    features.append(1 if atom.GetIsAromatic() else 0)
    features += one_hot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    features.append(1 if atom.IsInRing() else 0)
    return features

def get_bond_features(bond):
    bond_type = bond.GetBondType()
    features = [
        1 if bond_type == Chem.rdchem.BondType.SINGLE else 0,
        1 if bond_type == Chem.rdchem.BondType.DOUBLE else 0,
        1 if bond_type == Chem.rdchem.BondType.TRIPLE else 0,
        1 if bond_type == Chem.rdchem.BondType.AROMATIC else 0
    ]
    return features

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Add hydrogens
    mol = Chem.AddHs(mol)
    
    # Get atom features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))
    atom_features = torch.tensor(atom_features, dtype=torch.float)
    
    # Get edge indices and bond features
    edge_indices = []
    edge_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.append([i, j])
        edge_indices.append([j, i])
        edge_features.append(get_bond_features(bond))
        edge_features.append(get_bond_features(bond))
    
    if len(edge_indices) == 0:
        return None
        
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float)
    
    return Data(x=atom_features, edge_index=edge_index, edge_attr=edge_attr)

# Convert all molecules to graphs
print("Converting molecules to graphs...")
graphs = []
affinities = []
failed = 0

for item in data_list:
    graph = smiles_to_graph(item['smiles'])
    if graph is not None:
        graphs.append(graph)
        affinities.append(item['affinity'])
    else:
        failed += 1

print(f"Successfully converted: {len(graphs)}, Failed: {failed}")
affinities = np.array(affinities)

# Check feature dimensions
print(f"Atom feature dim: {graphs[0].x.shape[1]}")
print(f"Edge feature dim: {graphs[0].edge_attr.shape[1]}")

# Train/val/test split
print("\nSplitting data...")
indices = np.arange(len(graphs))
train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

# Create datasets
train_data = [graphs[i] for i in train_idx]
val_data = [graphs[i] for i in val_idx]
test_data = [graphs[i] for i in test_idx]

train_y = torch.tensor(affinities[train_idx], dtype=torch.float)
val_y = torch.tensor(affinities[val_idx], dtype=torch.float)
test_y = torch.tensor(affinities[test_idx], dtype=torch.float)

# Add y to graphs
for i, idx in enumerate(train_idx):
    graphs[idx].y = train_y[i]
for i, idx in enumerate(val_idx):
    graphs[idx].y = val_y[i]
for i, idx in enumerate(test_idx):
    graphs[idx].y = test_y[i]

# Create dataloaders
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64)
test_loader = DataLoader(test_data, batch_size=64)

# Define GNN model
class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels=256, out_channels=1):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, out_channels)
        
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x.squeeze()

# Initialize model
device = torch.device('cpu')
model = GNN(in_channels=graphs[0].x.shape[1], hidden_channels=256).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training function
def train_epoch(loader, model, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(loader, model):
    model.eval()
    preds = []
    true_vals = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            preds.extend(out.cpu().numpy())
            true_vals.extend(data.y.cpu().numpy())
    return np.array(preds), np.array(true_vals)

# Training loop with early stopping
print("\nTraining...")
best_val_loss = float('inf')
patience = 10
patience_counter = 0
best_model_state = None

for epoch in range(1, 51):
    train_loss = train_epoch(train_loader, model, optimizer, criterion)
    val_preds, val_true = evaluate(val_loader, model)
    val_loss = criterion(torch.tensor(val_preds), torch.tensor(val_true)).item()
    
    print(f"Epoch {epoch:3d}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict().copy()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

# Load best model
model.load_state_dict(best_model_state)

# Evaluate on test set
print("\nEvaluating on test set...")
test_preds, test_true = evaluate(test_loader, model)

# Calculate metrics
pearson_r, _ = pearsonr(test_preds, test_true)
mae = mean_absolute_error(test_true, test_preds)
rmse = np.sqrt(np.mean((test_preds - test_true) ** 2))

print(f"\n=== Test Set Results ===")
print(f"Pearson R: {pearson_r:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"Ridge+ECFP R: 0.614 (baseline)")

# Save model
torch.save(best_model_state, 'WORK_DIR / geock_gnn_model.pt')
print("\nModel saved to geock_gnn_model.pt")

# Save predictions
with open('WORK_DIR / results_gnn.tsv', 'w') as f:
    f.write("prediction\tactual\n")
    for pred, true in zip(test_preds, test_true):
        f.write(f"{pred:.4f}\t{true:.4f}")
print("Predictions saved to results_gnn.tsv")
