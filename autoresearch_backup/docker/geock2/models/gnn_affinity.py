"""
gnn_affinity.py - PocketGNN + MLP for Binding Affinity Prediction

Architecture:
1. EdgeConv layers to encode receptor pocket atoms
2. Global pooling to get pocket embedding
3. MLP to combine pocket embedding with pose features
4. Output: binding affinity (ΔG in kcal/mol)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class GNNConfig:
    """Configuration for PocketGNN model"""
    # GNN parameters
    node_features: int = 14  # Atomic number + other features
    hidden_dim: int = 32
    num_edge_conv_layers: int = 2
    k_neighbors: int = 8  # k for k-NN in EdgeConv
    
    # MLP parameters
    pose_features: int = 24  # 24D pose vector
    physics_features: int = 5  # Vina score components
    mlp_hidden: int = 64
    mlp_layers: int = 2
    dropout: float = 0.3
    
    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4


def atom_features_to_one_hot(atom_types: List[str], max_atomic_num: int = 100) -> torch.Tensor:
    """Convert atom types to one-hot features"""
    ATOMIC_MAP = {
        'H': 1, 'C': 6, 'N': 7, 'O': 8, 'S': 16, 'P': 15,
        'F': 9, 'Cl': 17, 'Br': 35, 'I': 53,
        'Fe': 26, 'Zn': 30, 'Ca': 20, 'Mg': 12, 'Mn': 25
    }
    
    features = []
    for atom_type in atom_types:
        atomic_num = ATOMIC_MAP.get(atom_type, 6)  # Default to carbon
        one_hot = torch.zeros(max_atomic_num)
        one_hot[atomic_num] = 1.0
        
        # Additional features
        is_hydrophobic = 1.0 if atom_type in {'C', 'S'} else 0.0
        is_hbond_donor = 1.0 if atom_type in {'N', 'O'} else 0.0
        is_hbond_acceptor = 1.0 if atom_type in {'N', 'O', 'S'} else 0.0
        is_aromatic = 0.0  # Would need ring detection
        
        feat = torch.cat([one_hot, torch.tensor([
            is_hydrophobic, is_hbond_donor, is_hbond_acceptor, is_aromatic
        ])])
        features.append(feat)
    
    return torch.stack(features)


class EdgeConvLayer(nn.Module):
    """
    EdgeConv layer for graph neural networks.
    Uses k-nearest neighbors to compute edge features.
    """
    
    def __init__(self, in_features: int, out_features: int, k: int = 16):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Linear(in_features * 2, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [N, in_features]
            edge_index: Edge indices [2, E]
        
        Returns:
            Updated node features [N, out_features]
        """
        N = x.shape[0]
        
        # Compute pairwise distances if no edge_index
        if edge_index is None or edge_index.numel() == 0:
            # Simple pooling: mean + std across atoms
            in_feat = x.shape[1]
            mean_feat = x.mean(dim=0, keepdim=True)  # [1, in_features]
            std_feat = x.std(dim=0, keepdim=True)   # [1, in_features]
            pooled = torch.cat([mean_feat, std_feat], dim=-1)  # [1, 2*in_features]
            
            # Project to output dimension
            proj = torch.nn.Linear(in_feat * 2, self.conv[0].out_features, bias=False).to(x.device)
            x_out = proj(pooled)  # [1, out_features]
        else:
            # Use provided edges
            x_i = x[edge_index[0]]  # [E, F]
            x_j = x[edge_index[1]]  # [E, F]
            
            # Edge features: concatenation of node features + difference
            edge_feat = torch.cat([x_i, x_j - x_i], dim=-1)  # [E, 2F]
            
            # Apply MLP
            edge_feat = self.conv(edge_feat)  # [E, out_features]
            
            # Aggregate to nodes (max pooling)
            out_feat = self.conv[0].out_features
            x_out = torch.zeros(N, out_feat, device=x.device)
            x_out.index_add_(0, edge_index[0], edge_feat)
            
            # Normalize by degree (approximate)
            deg = torch.zeros(N, device=x.device)
            deg.index_add_(0, edge_index[0], torch.ones(edge_index.shape[1], device=x.device))
            deg = torch.clamp(deg, min=1)
            x_out = x_out / deg.unsqueeze(-1)
        
        return x_out


class PocketGNN(nn.Module):
    """
    Graph Neural Network for encoding receptor pocket atoms.
    
    Uses EdgeConv layers to learn spatial patterns in the binding site.
    """
    
    def __init__(self, config: GNNConfig):
        super().__init__()
        self.config = config
        
        # EdgeConv layers
        self.edge_convs = nn.ModuleList()
        in_feat = config.node_features
        
        for i in range(config.num_edge_conv_layers):
            out_feat = config.hidden_dim if i < config.num_edge_conv_layers - 1 else config.hidden_dim
            self.edge_convs.append(EdgeConvLayer(in_feat, out_feat, k=config.k_neighbors))
            in_feat = out_feat
        
        # Global pooling (attention-based)
        self.pool = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.Tanh()
        )
    
    def forward(self, x: torch.Tensor, batch_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Node features [N, node_features]
            batch_idx: Batch indices for pooling [N] (optional)
        
        Returns:
            Pocket embedding [hidden_dim] or [batch_size, hidden_dim]
        """
        # Apply EdgeConv layers
        for conv in self.edge_convs:
            x = conv(x, torch.tensor([], device=x.device).long())
            x = F.relu(x)
        
        # Attention-based pooling
        attn = self.pool(x)  # [N, hidden_dim//2]
        
        if batch_idx is None:
            # Single graph - mean pooling with attention
            attn_weights = F.softmax(attn.sum(-1, keepdim=True), dim=0)
            embedding = (x * attn_weights).sum(dim=0)  # [hidden_dim]
        else:
            # Multiple graphs - pool per graph
            embedding = []
            for i in range(batch_idx.max().item() + 1):
                mask = batch_idx == i
                if mask.sum() > 0:
                    g_x = x[mask]
                    g_attn = attn[mask]
                    w = F.softmax(g_attn.sum(-1, keepdim=True), dim=0)
                    emb = (g_x * w).sum(dim=0)
                    embedding.append(emb)
            embedding = torch.stack(embedding) if embedding else torch.zeros(1, self.config.hidden_dim, device=x.device)
        
        return embedding


class AffinityMLP(nn.Module):
    """
    MLP for combining pocket embedding with pose features and scoring.
    """
    
    def __init__(self, config: GNNConfig):
        super().__init__()
        self.config = config
        
        total_features = config.hidden_dim + config.pose_features + config.physics_features
        
        # Build MLP layers
        layers = []
        in_dim = total_features
        
        for i in range(config.mlp_layers):
            out_dim = config.mlp_hidden if i < config.mlp_layers - 1 else 1
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim) if out_dim > 1 else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(config.dropout)
            ])
            in_dim = out_dim
        
        # Remove last ReLU and dropout for output layer
        layers = layers[:-2] if config.mlp_layers > 1 else [nn.Linear(total_features, 1)]
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, pocket_emb: torch.Tensor, pose_feat: torch.Tensor, 
                physics_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pocket_emb: Pocket embedding [hidden_dim] or [batch, hidden_dim]
            pose_feat: Pose features [pose_features] or [batch, pose_features]
            physics_feat: Physics scoring features [physics_features] or [batch, physics_features]
        
        Returns:
            Predicted binding affinity [1] or [batch, 1]
        """
        # Concatenate all features
        if pocket_emb.dim() == 1:
            pocket_emb = pocket_emb.unsqueeze(0)
        if pose_feat.dim() == 1:
            pose_feat = pose_feat.unsqueeze(0)
        if physics_feat.dim() == 1:
            physics_feat = physics_feat.unsqueeze(0)
        
        x = torch.cat([pocket_emb, pose_feat, physics_feat], dim=-1)
        
        # Apply MLP
        out = self.mlp(x)
        
        return out.squeeze(-1)


class PocketEncoder(nn.Module):
    """Encodes receptor pocket to fixed-size embedding using mean pooling"""
    
    def __init__(self, config: GNNConfig):
        super().__init__()
        self.config = config
        
        self.proj = nn.Sequential(
            nn.Linear(config.node_features, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU()
        )
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, pocket_features: torch.Tensor) -> torch.Tensor:
        mean_feat = pocket_features.mean(dim=0, keepdim=True)
        x = self.proj(mean_feat)
        return x


class AffinityPredictor(nn.Module):
    """Predicts affinity from pocket embedding + pose + physics features"""
    
    def __init__(self, config: GNNConfig):
        super().__init__()
        self.config = config
        
        total_features = config.hidden_dim + config.pose_features + config.physics_features
        
        self.net = nn.Sequential(
            nn.Linear(total_features, config.mlp_hidden),
            nn.LayerNorm(config.mlp_hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_hidden, config.mlp_hidden // 2),
            nn.LayerNorm(config.mlp_hidden // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_hidden // 2, 1)
        )
    
    def forward(self, pocket_emb: torch.Tensor, pose_feat: torch.Tensor, 
                physics_feat: torch.Tensor) -> torch.Tensor:
        if pocket_emb.dim() == 1:
            pocket_emb = pocket_emb.unsqueeze(0)
        if pose_feat.dim() == 1:
            pose_feat = pose_feat.unsqueeze(0)
        if physics_feat.dim() == 1:
            physics_feat = physics_feat.unsqueeze(0)
        
        # Handle extra dimension in pocket_emb
        if pocket_emb.dim() == 3 and pocket_emb.shape[1] == 1:
            pocket_emb = pocket_emb.squeeze(1)
        
        x = torch.cat([pocket_emb, pose_feat, physics_feat], dim=-1)
        return self.net(x).squeeze(-1)


class PocketGNNAffinity(nn.Module):
    """
    Complete model: PocketGNN Encoder + Affinity Predictor
    
    Two-stage architecture:
    1. Encode receptor pocket once to get embedding
    2. Predict affinity from embedding + pose + physics
    """
    
    def __init__(self, config: Optional[GNNConfig] = None):
        super().__init__()
        self.config = config or GNNConfig()
        
        self.encoder = PocketEncoder(self.config)
        self.predictor = AffinityPredictor(self.config)
        
        self.pose_mean = nn.Parameter(torch.zeros(self.config.pose_features))
        self.pose_std = nn.Parameter(torch.ones(self.config.pose_features))
        self.physics_mean = nn.Parameter(torch.zeros(self.config.physics_features))
        self.physics_std = nn.Parameter(torch.ones(self.config.physics_features))
    
    def encode_pocket(self, pocket_features: torch.Tensor) -> torch.Tensor:
        """Encode pocket to embedding
        
        Args:
            pocket_features: [N, features]
        
        Returns:
            embedding [1, hidden_dim]
        """
        return self.encoder(pocket_features)
    
    def forward(self, pocket_features: torch.Tensor, pose_features: torch.Tensor,
                physics_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pocket_features: [N, node_features] receptor atom features (same for batch)
            pose_features: [batch, pose_features]
            physics_features: [batch, physics_features]
        
        Returns:
            Predicted affinity [batch]
        """
        # Encode pocket once
        pocket_emb = self.encoder(pocket_features)
        
        # Expand pocket embedding to match batch size
        batch_size = pose_features.shape[0]
        if pocket_emb.dim() == 1:
            pocket_emb = pocket_emb.unsqueeze(0).expand(batch_size, -1)
        
        # Normalize features
        pose_norm = (pose_features - self.pose_mean) / (self.pose_std + 1e-8)
        physics_norm = (physics_features - self.physics_mean) / (self.physics_std + 1e-8)
        
        # Predict affinity
        affinity = self.predictor(pocket_emb, pose_norm, physics_norm)
        
        return affinity
    
    def get_embedding(self, pocket_features: torch.Tensor) -> torch.Tensor:
        """Get pocket embedding for analysis"""
        return self.encoder(pocket_features)


def create_pocketgnn_model(config: Optional[GNNConfig] = None) -> PocketGNNAffinity:
    """Factory function to create model"""
    return PocketGNNAffinity(config)


# Example usage and testing
if __name__ == "__main__":
    # Test model
    config = GNNConfig(
        node_features=14,
        hidden_dim=64,
        num_edge_conv_layers=3,
        pose_features=24,
        physics_features=5
    )
    
    model = create_pocketgnn_model(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 4
    n_atoms = 50
    
    pocket_features = torch.randn(n_atoms, config.node_features)
    pose_features = torch.randn(batch_size, config.pose_features)
    physics_features = torch.randn(batch_size, config.physics_features)
    
    with torch.no_grad():
        output = model(pocket_features, pose_features, physics_features)
    
    print(f"Output shape: {output.shape}")
    print(f"Output (first 3): {output[:3]}")
    
    # Test on CPU (should work)
    print("\nModel created successfully!")
