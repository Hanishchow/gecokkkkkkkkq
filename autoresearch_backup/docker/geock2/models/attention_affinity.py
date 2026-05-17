"""
attention_affinity.py - AttentionVAE + MLP for Binding Affinity Prediction

Architecture:
1. Multi-head self-attention layers to encode receptor pocket atoms
2. VAE latent space for variational inference
3. MLP to combine latent embedding with pose features
4. Output: binding affinity (ΔG in kcal/mol)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class AttentionVAEConfig:
    """Configuration for AttentionVAE model"""
    # Atom features
    node_features: int = 18  # One-hot + chemistry features
    
    # Transformer parameters
    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 3
    dropout: float = 0.1
    
    # VAE parameters
    latent_dim: int = 32
    
    # MLP parameters
    pose_features: int = 24
    physics_features: int = 5
    mlp_hidden: int = 128
    mlp_layers: int = 3
    
    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            pe = torch.zeros(1, seq_len, self.d_model, device=x.device)
            position = torch.arange(0, seq_len, dtype=torch.float, device=x.device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.d_model, 2, device=x.device).float() * (-torch.log(torch.tensor(10000.0, device=x.device)) / self.d_model))
            pe[:, :, 0::2] = torch.sin(position * div_term)
            pe[:, :, 1::2] = torch.cos(position * div_term)
            x = x + pe
        else:
            x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer with multi-head attention"""
    
    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x2 = self.norm1(x)
        x2, _ = self.self_attn(x2, x2, x2, attn_mask=mask)
        x = x + self.dropout1(x2)
        
        x2 = self.norm2(x)
        x2 = self.linear2(self.dropout(F.relu(self.linear1(x2))))
        x = x + self.dropout2(x2)
        
        return x


class PocketTransformerVAE(nn.Module):
    """
    Transformer-based encoder with VAE latent space.
    
    Uses multi-head self-attention to encode receptor pocket atoms
    into a variational latent space.
    """
    
    def __init__(self, config: AttentionVAEConfig):
        super().__init__()
        self.config = config
        
        self.input_proj = nn.Linear(config.node_features, config.hidden_dim)
        self.pos_enc = PositionalEncoding(config.hidden_dim, dropout=config.dropout)
        
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(
                config.hidden_dim,
                config.num_heads,
                config.hidden_dim * 2,
                config.dropout
            )
            for _ in range(config.num_layers)
        ])
        
        self.norm = nn.LayerNorm(config.hidden_dim)
        
        self.fc_mu = nn.Linear(config.hidden_dim, config.latent_dim)
        self.fc_logvar = nn.Linear(config.hidden_dim, config.latent_dim)
        
        self.attention_pool = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(config.hidden_dim // 2, 1)
        )
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu
    
    def forward(self, x: torch.Tensor, return_latent: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N = x.shape[0]
        
        x = self.input_proj(x)
        
        x = x.unsqueeze(1)
        x = self.pos_enc(x)
        
        for layer in self.transformer_layers:
            x = layer(x)
        
        x = self.norm(x)
        
        attn_weights = self.attention_pool(x)
        attn_weights = F.softmax(attn_weights.squeeze(-1), dim=0).unsqueeze(-1)
        pooled = (x * attn_weights).sum(dim=0)
        
        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)
        z = self.reparameterize(mu, logvar)
        
        if return_latent:
            return z, mu, logvar
        return z, mu, logvar
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z, _, _ = self.forward(x, return_latent=False)
        return z


class VAEEncoder(nn.Module):
    """Transformer encoder for receptor pocket with VAE"""
    
    def __init__(self, config: AttentionVAEConfig):
        super().__init__()
        self.config = config
        self.transformer = PocketTransformerVAE(config)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.transformer(x, return_latent=True)


class AffinityVAEDecoder(nn.Module):
    """Decoder for VAE latent + pose + physics"""
    
    def __init__(self, config: AttentionVAEConfig):
        super().__init__()
        self.config = config
        
        total_features = config.latent_dim + config.pose_features + config.physics_features
        
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
    
    def forward(self, latent: torch.Tensor, pose_feat: torch.Tensor, 
                physics_feat: torch.Tensor) -> torch.Tensor:
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
        if pose_feat.dim() == 1:
            pose_feat = pose_feat.unsqueeze(0)
        if physics_feat.dim() == 1:
            physics_feat = physics_feat.unsqueeze(0)
        
        x = torch.cat([latent, pose_feat, physics_feat], dim=-1)
        return self.net(x).squeeze(-1)


class AttentionVAEAffinity(nn.Module):
    """
    Complete model: AttentionVAE Encoder + Affinity Decoder
    
    VAE architecture with:
    - Transformer encoder for receptor pocket
    - Variational latent space
    - MLP decoder for binding affinity prediction
    """
    
    def __init__(self, config: Optional[AttentionVAEConfig] = None):
        super().__init__()
        self.config = config or AttentionVAEConfig()
        
        self.encoder = VAEEncoder(self.config)
        self.decoder = AffinityVAEDecoder(self.config)
        
        self.pose_mean = nn.Parameter(torch.zeros(self.config.pose_features))
        self.pose_std = nn.Parameter(torch.ones(self.config.pose_features))
        self.physics_mean = nn.Parameter(torch.zeros(self.config.physics_features))
        self.physics_std = nn.Parameter(torch.ones(self.config.physics_features))
    
    def encode(self, pocket_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode pocket to latent distribution"""
        return self.encoder(pocket_features)
    
    def forward(self, pocket_features: torch.Tensor, pose_features: torch.Tensor,
                physics_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            pocket_features: [N, node_features] receptor atom features
            pose_features: [batch, pose_features]
            physics_features: [batch, physics_features]
        
        Returns:
            (affinity, mu, logvar)
        """
        z, mu, logvar = self.encoder(pocket_features)
        
        # Expand latent to match batch size
        batch_size = pose_features.shape[0]
        if z.shape[0] == 1:
            z = z.expand(batch_size, -1)
        
        # Normalize features
        pose_norm = (pose_features - self.pose_mean) / (self.pose_std + 1e-8)
        physics_norm = (physics_features - self.physics_mean) / (self.physics_std + 1e-8)
        
        affinity = self.decoder(z, pose_norm, physics_norm)
        
        return affinity, mu, logvar
    
    def predict(self, pocket_features: torch.Tensor, pose_features: torch.Tensor,
                physics_features: torch.Tensor) -> torch.Tensor:
        """Prediction only (no VAE sampling)"""
        with torch.no_grad():
            return self(pocket_features, pose_features, physics_features)[0]
    
    def vae_loss(self, affinity_pred: torch.Tensor, affinity_true: torch.Tensor,
                 mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        """
        VAE loss = MSE affinity loss + beta * KL divergence
        """
        mse_loss = F.mse_loss(affinity_pred, affinity_true)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return mse_loss + beta * kl_loss
    
    def get_embedding(self, pocket_features: torch.Tensor) -> torch.Tensor:
        """Get latent embedding for analysis"""
        z, _, _ = self.encoder(pocket_features)
        return z


def create_attentionvae_model(config: Optional[AttentionVAEConfig] = None) -> AttentionVAEAffinity:
    """Factory function to create model"""
    return AttentionVAEAffinity(config)


def create_hybrid_model(gnn_config: Optional['GNNConfig'] = None, 
                        vae_config: Optional[AttentionVAEConfig] = None,
                        fusion_dim: int = 128) -> nn.Module:
    """
    Create hybrid model combining PocketGNN and AttentionVAE.
    
    Fusion strategies:
    1. Concatenate both embeddings
    2. Learn attention weights over both
    """
    from .gnn_affinity import PocketGNNAffinity, GNNConfig
    
    gnn_config = gnn_config or GNNConfig()
    vae_config = vae_config or AttentionVAEConfig()
    
    class HybridModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.gnn = PocketGNNAffinity(gnn_config)
            self.vae = AttentionVAEAffinity(vae_config)
            
            gnn_dim = gnn_config.hidden_dim
            vae_dim = vae_config.latent_dim
            
            self.fusion = nn.Sequential(
                nn.Linear(gnn_dim + vae_dim, fusion_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(fusion_dim, 1)
            )
        
        def forward(self, pocket_features, pose_features, physics_features):
            gnn_emb = self.gnn.pocket_gnn(pocket_features)
            vae_emb, mu, logvar = self.vae.encoder(pocket_features, return_latent=True)
            
            combined = torch.cat([gnn_emb, vae_emb], dim=-1)
            return self.fusion(combined), mu, logvar
    
    return HybridModel()


if __name__ == "__main__":
    config = AttentionVAEConfig(
        node_features=18,
        hidden_dim=128,
        num_heads=4,
        num_layers=3,
        latent_dim=32,
        pose_features=24,
        physics_features=5
    )
    
    model = create_attentionvae_model(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    batch_size = 4
    n_atoms = 50
    
    pocket_features = torch.randn(n_atoms, config.node_features)
    pose_features = torch.randn(batch_size, config.pose_features)
    physics_features = torch.randn(batch_size, config.physics_features)
    
    affinity, mu, logvar = model(pocket_features, pose_features, physics_features)
    print(f"Affinity shape: {affinity.shape}")
    print(f"Latent mu shape: {mu.shape}")
    print(f"Latent logvar shape: {logvar.shape}")
    
    test_affinity = torch.randn(batch_size)
    loss = model.vae_loss(affinity, test_affinity, mu, logvar, beta=0.1)
    print(f"VAE loss: {loss.item():.4f}")
    
    print("\nAttentionVAE model created successfully!")
