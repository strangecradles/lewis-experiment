"""
Cross-attention connector architecture for Lewis superadditivity experiment.

Key components:
- CrossAttentionConnector: bidirectional cross-attention between two models
- ConnectorBank: manages all pairwise connectors for a model subset
- TaskHead: maps concatenated CLS tokens to answer logits
- ComposedSystem: full forward pass combining models + connectors + task head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
import math
from itertools import combinations

from .models import ModelBank


@dataclass
class ConnectorConfig:
    """Configuration for cross-attention connectors."""
    hidden_dim: int = 256
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    

class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross-attention module."""
    
    def __init__(self, query_dim: int, key_dim: int, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Projections for queries, keys, values
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(key_dim, hidden_dim)
        self.value_proj = nn.Linear(key_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, query_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: [batch, seq_len_q, query_dim]
            key: [batch, seq_len_k, key_dim] 
            value: [batch, seq_len_k, key_dim]
            
        Returns:
            output: [batch, seq_len_q, query_dim]
        """
        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.shape[1]
        
        # Project to hidden dimension
        Q = self.query_proj(query)  # [batch, seq_len_q, hidden_dim]
        K = self.key_proj(key)      # [batch, seq_len_k, hidden_dim]  
        V = self.value_proj(value)  # [batch, seq_len_k, hidden_dim]
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, heads, seq_q, head_dim]
        K = K.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, heads, seq_k, head_dim]
        V = V.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, heads, seq_k, head_dim]
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [batch, heads, seq_q, seq_k]
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # [batch, heads, seq_q, head_dim]
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.hidden_dim
        )  # [batch, seq_q, hidden_dim]
        
        # Final projection
        output = self.out_proj(attn_output)  # [batch, seq_q, query_dim]
        
        return output


class CrossAttentionLayer(nn.Module):
    """Single cross-attention layer with residual connection and layer norm."""
    
    def __init__(self, query_dim: int, key_dim: int, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = MultiHeadCrossAttention(query_dim, key_dim, hidden_dim, num_heads, dropout)
        self.layer_norm = nn.LayerNorm(query_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Apply cross-attention with residual connection."""
        # Cross-attention
        attn_output = self.cross_attn(query, key, value)
        
        # Residual connection and layer norm
        output = self.layer_norm(query + self.dropout(attn_output))
        
        return output


class CrossAttentionConnector(nn.Module):
    """
    Bidirectional cross-attention connector between two models.
    
    Takes patch tokens from two models and enriches their CLS tokens through 
    cross-model attention. ~5M parameters per connector.
    """
    
    def __init__(self, 
                 model_i_dim: int, 
                 model_j_dim: int, 
                 config: ConnectorConfig = ConnectorConfig()):
        super().__init__()
        self.config = config
        self.model_i_dim = model_i_dim
        self.model_j_dim = model_j_dim
        
        # Cross-attention from i to j (i queries j)
        self.i_to_j_layers = nn.ModuleList([
            CrossAttentionLayer(
                query_dim=model_i_dim,
                key_dim=model_j_dim, 
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                dropout=config.dropout
            )
            for _ in range(config.num_layers)
        ])
        
        # Cross-attention from j to i (j queries i)
        self.j_to_i_layers = nn.ModuleList([
            CrossAttentionLayer(
                query_dim=model_j_dim,
                key_dim=model_i_dim,
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                dropout=config.dropout
            )
            for _ in range(config.num_layers)
        ])
        
        # Final projections to extract enriched CLS tokens
        self.i_cls_proj = nn.Linear(model_i_dim, model_i_dim)
        self.j_cls_proj = nn.Linear(model_j_dim, model_j_dim)
        
    def forward(self, 
                i_features: Dict[str, torch.Tensor], 
                j_features: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            i_features: {'patch_tokens': [batch, num_patches_i, dim_i], 'cls_token': [batch, dim_i]}
            j_features: {'patch_tokens': [batch, num_patches_j, dim_j], 'cls_token': [batch, dim_j]}
            
        Returns:
            Tuple of enriched CLS tokens (i_cls_enriched, j_cls_enriched)
        """
        i_patches = i_features['patch_tokens']  # [batch, num_patches_i, dim_i]
        j_patches = j_features['patch_tokens']  # [batch, num_patches_j, dim_j]
        i_cls = i_features['cls_token'].unsqueeze(1)  # [batch, 1, dim_i]
        j_cls = j_features['cls_token'].unsqueeze(1)  # [batch, 1, dim_j]
        
        # Cross-attention: i CLS token attends to j patches
        i_enriched = i_cls
        for layer in self.i_to_j_layers:
            i_enriched = layer(query=i_enriched, key=j_patches, value=j_patches)
        
        # Cross-attention: j CLS token attends to i patches  
        j_enriched = j_cls
        for layer in self.j_to_i_layers:
            j_enriched = layer(query=j_enriched, key=i_patches, value=i_patches)
        
        # Final projections and squeeze
        i_cls_enriched = self.i_cls_proj(i_enriched.squeeze(1))  # [batch, dim_i]
        j_cls_enriched = self.j_cls_proj(j_enriched.squeeze(1))  # [batch, dim_j]
        
        return i_cls_enriched, j_cls_enriched
    
    def num_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ConnectorBank(nn.Module):
    """
    Manages all pairwise connectors for a given model subset.
    
    For subset {A, B, C}, creates connectors: C_AB, C_AC, C_BC
    """
    
    def __init__(self, 
                 model_names: List[str], 
                 model_dims: Dict[str, int], 
                 config: ConnectorConfig = ConnectorConfig()):
        super().__init__()
        self.model_names = sorted(model_names)  # Ensure consistent ordering
        self.model_dims = model_dims
        self.config = config
        
        # Create all pairwise connectors
        self.connectors = nn.ModuleDict()
        self.pairs = list(combinations(self.model_names, 2))
        
        for model_i, model_j in self.pairs:
            connector_name = f"{model_i}_{model_j}"
            self.connectors[connector_name] = CrossAttentionConnector(
                model_i_dim=model_dims[model_i],
                model_j_dim=model_dims[model_j],
                config=config
            )
    
    def forward(self, features: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Apply all pairwise connectors and return enriched CLS tokens.
        
        Args:
            features: Dict mapping model_name -> {'patch_tokens': Tensor, 'cls_token': Tensor}
            
        Returns:
            Dict mapping model_name -> enriched_cls_token
        """
        # Initialize enriched CLS tokens with original values
        enriched_cls = {name: features[name]['cls_token'] for name in self.model_names}
        
        # Track how many times each model's CLS token gets updated
        update_counts = {name: 0 for name in self.model_names}
        accumulated_updates = {name: 0 for name in self.model_names}
        
        # Apply all pairwise connectors
        for model_i, model_j in self.pairs:
            connector_name = f"{model_i}_{model_j}"
            connector = self.connectors[connector_name]
            
            # Get enriched CLS tokens from this connector
            i_enriched, j_enriched = connector(features[model_i], features[model_j])
            
            # Accumulate updates (will average later)
            accumulated_updates[model_i] += i_enriched
            accumulated_updates[model_j] += j_enriched
            update_counts[model_i] += 1
            update_counts[model_j] += 1
        
        # Average the updates from different connectors
        for name in self.model_names:
            if update_counts[name] > 0:
                enriched_cls[name] = accumulated_updates[name] / update_counts[name]
        
        return enriched_cls
    
    def get_connector_info(self) -> Dict[str, int]:
        """Get parameter counts for each connector."""
        info = {}
        for name, connector in self.connectors.items():
            info[name] = connector.num_parameters()
        return info


class TaskHead(nn.Module):
    """
    Task-specific head that maps concatenated CLS tokens to answer logits.
    
    2-layer MLP with GELU activation, hidden dimension 512.
    """
    
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, concatenated_cls: torch.Tensor) -> torch.Tensor:
        """
        Args:
            concatenated_cls: [batch, total_cls_dim] - concatenated CLS tokens
            
        Returns:
            logits: [batch, num_classes]
        """
        return self.mlp(concatenated_cls)
    
    def num_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ComposedSystem(nn.Module):
    """
    Complete system combining ModelBank + ConnectorBank + TaskHead.
    
    Implements the full forward pass from images to answer logits.
    """
    
    def __init__(self, 
                 model_bank: ModelBank,
                 active_models: List[str],
                 num_classes: int,
                 connector_config: ConnectorConfig = ConnectorConfig()):
        super().__init__()
        self.model_bank = model_bank
        self.active_models = sorted(active_models)
        self.num_classes = num_classes
        
        # Get model dimensions
        model_info = model_bank.get_model_info()
        self.model_dims = {
            name: model_info[name]['config'].embed_dim 
            for name in self.active_models
        }
        
        # Create connector bank (only if we have multiple models)
        if len(self.active_models) > 1:
            self.connector_bank = ConnectorBank(
                model_names=self.active_models,
                model_dims=self.model_dims,
                config=connector_config
            )
        else:
            self.connector_bank = None
        
        # Create task head
        total_cls_dim = sum(self.model_dims.values())
        self.task_head = TaskHead(
            input_dim=total_cls_dim,
            num_classes=num_classes
        )
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass from images to answer logits.
        
        Args:
            images: [batch_size, 3, H, W]
            
        Returns:
            logits: [batch_size, num_classes]
        """
        # Extract features from active models (frozen)
        features = self.model_bank.get_features(images, self.active_models)
        
        # Apply connectors if multiple models
        if self.connector_bank is not None:
            enriched_cls = self.connector_bank(features)
        else:
            # Single model - just use original CLS token
            enriched_cls = {
                name: features[name]['cls_token'] 
                for name in self.active_models
            }
        
        # Concatenate CLS tokens in consistent order
        cls_tokens = [enriched_cls[name] for name in self.active_models]
        concatenated_cls = torch.cat(cls_tokens, dim=1)  # [batch, total_cls_dim]
        
        # Apply task head
        logits = self.task_head(concatenated_cls)
        
        return logits
    
    def get_trainable_parameters(self) -> Dict[str, int]:
        """Get count of trainable parameters by component."""
        params = {'task_head': self.task_head.num_parameters()}
        
        if self.connector_bank is not None:
            connector_info = self.connector_bank.get_connector_info()
            params.update(connector_info)
            params['total_connectors'] = sum(connector_info.values())
        
        params['total_trainable'] = sum(params.values())
        return params
    
    def get_system_info(self) -> Dict[str, any]:
        """Get comprehensive system information."""
        info = {
            'active_models': self.active_models,
            'model_dims': self.model_dims,
            'num_classes': self.num_classes,
            'trainable_params': self.get_trainable_parameters()
        }
        
        if self.connector_bank is not None:
            info['num_connectors'] = len(self.connector_bank.pairs)
            info['connector_pairs'] = self.connector_bank.pairs
        else:
            info['num_connectors'] = 0
            info['connector_pairs'] = []
        
        return info


def test_connectors():
    """Test the connector architecture."""
    print("Testing connector architecture...")
    
    # Create mock model bank
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_bank = ModelBank(device=device)
    
    # Test different model combinations
    test_cases = [
        ['dinov2'],           # Single model
        ['dinov2', 'siglip'], # Pair
        ['dinov2', 'siglip', 'mae']  # All three
    ]
    
    batch_size = 2
    num_classes = 1000
    dummy_images = torch.randn(batch_size, 3, 224, 224).to(device)
    
    for active_models in test_cases:
        print(f"\nTesting with models: {active_models}")
        
        # Create composed system
        system = ComposedSystem(
            model_bank=model_bank,
            active_models=active_models,
            num_classes=num_classes
        )
        system = system.to(device)
        
        # Forward pass
        logits = system(dummy_images)
        print(f"Output shape: {logits.shape}")
        
        # Print system info
        info = system.get_system_info()
        print(f"Trainable parameters: {info['trainable_params']}")
        
        # Test gradient flow
        loss = F.cross_entropy(logits, torch.randint(0, num_classes, (batch_size,)).to(device))
        loss.backward()
        
        # Check that only connectors and task head have gradients
        has_grads = []
        for name, param in system.named_parameters():
            if param.grad is not None:
                has_grads.append(name)
        
        print(f"Parameters with gradients: {len(has_grads)}")
        assert all('model_bank' not in name for name in has_grads), "Model bank should be frozen!"
        
        # Clear gradients for next test
        system.zero_grad()
    
    print("\nConnector tests passed!")


if __name__ == "__main__":
    test_connectors()