"""
positional_encoding_experiment.py — PatchCore Spatial Positional Encoding Experiment
--------------------------------------------------------------------------------------
Provides isolated 2D sinusoidal positional encoding generation and feature augmentation
for PatchCore patch feature maps.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def generate_2d_sinusoidal_positional_encoding(
    H: int,
    W: int,
    pos_dim: int = 16,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Generates a deterministic 2D sinusoidal positional encoding matrix of shape (H * W, pos_dim).
    
    Args:
        H: Feature map height (e.g. 32).
        W: Feature map width (e.g. 32).
        pos_dim: Total positional embedding dimension (must be even, split equally into pos_dim//2 for X and Y).
        device: PyTorch device (CPU or CUDA).
        
    Returns:
        Tensor of shape (H * W, pos_dim)
    """
    assert pos_dim % 2 == 0, f"pos_dim must be even, got {pos_dim}"
    dim_per_axis = pos_dim // 2

    # Normalized spatial coordinates in range [0, 1]
    y_coords = torch.linspace(0.0, 1.0, steps=H, device=device) if H > 1 else torch.zeros(1, device=device)
    x_coords = torch.linspace(0.0, 1.0, steps=W, device=device) if W > 1 else torch.zeros(1, device=device)

    # 2D Grid: (H, W)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")

    # Flatten coordinates: (H * W,)
    grid_y_flat = grid_y.reshape(-1)
    grid_x_flat = grid_x.reshape(-1)

    # Frequencies: 10000^(2i / dim_per_axis) for i = 0 .. dim_per_axis // 2 - 1
    num_freqs = dim_per_axis // 2
    freq_exponent = torch.arange(num_freqs, dtype=torch.float32, device=device) * (2.0 / dim_per_axis)
    inv_freq = 1.0 / (10000.0 ** freq_exponent)

    # Compute PE for X: (H*W, num_freqs)
    x_args = grid_x_flat.unsqueeze(1) * inv_freq.unsqueeze(0)
    pe_x = torch.cat([torch.sin(x_args), torch.cos(x_args)], dim=-1)

    # Compute PE for Y: (H*W, num_freqs)
    y_args = grid_y_flat.unsqueeze(1) * inv_freq.unsqueeze(0)
    pe_y = torch.cat([torch.sin(y_args), torch.cos(y_args)], dim=-1)

    # Concatenate X and Y encodings: (H*W, pos_dim)
    pe_2d = torch.cat([pe_x, pe_y], dim=-1)
    return pe_2d


def augment_patch_features_with_position(
    features: torch.Tensor,
    H: int,
    W: int,
    pos_dim: int = 16,
    gamma: float = 1.0
) -> torch.Tensor:
    """
    Appends 2D sinusoidal positional encoding to extracted patch visual features.
    
    Args:
        features: Tensor of shape (B, N, D) or (N, D) where N = H * W.
        H: Feature map height.
        W: Feature map width.
        pos_dim: Positional embedding dimension (default: 16).
        gamma: Spatial weighting hyperparameter for positional features.
        
    Returns:
        Tensor of shape (B, N, D + pos_dim) or (N, D + pos_dim).
    """
    is_unbatched = features.ndim == 2
    if is_unbatched:
        features = features.unsqueeze(0)  # (1, N, D)

    B, N, D = features.shape
    assert N == H * W, f"Feature count N={N} does not match H*W={H*W}"

    pe = generate_2d_sinusoidal_positional_encoding(H, W, pos_dim=pos_dim, device=features.device)
    pe_expanded = (pe * gamma).unsqueeze(0).expand(B, -1, -1)

    # Concatenate visual and positional features
    enhanced = torch.cat([features, pe_expanded], dim=-1)

    return enhanced.squeeze(0) if is_unbatched else enhanced
