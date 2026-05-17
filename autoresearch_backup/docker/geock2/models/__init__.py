"""
geock.models - Neural network models for binding affinity prediction
"""

from .gnn_affinity import (
    PocketGNNAffinity,
    GNNConfig,
    create_pocketgnn_model,
    atom_features_to_one_hot
)

from .attention_affinity import (
    AttentionVAEAffinity,
    AttentionVAEConfig,
    create_attentionvae_model,
    create_hybrid_model
)

__all__ = [
    'PocketGNNAffinity',
    'GNNConfig',
    'create_pocketgnn_model',
    'atom_features_to_one_hot',
    'AttentionVAEAffinity',
    'AttentionVAEConfig',
    'create_attentionvae_model',
    'create_hybrid_model'
]
