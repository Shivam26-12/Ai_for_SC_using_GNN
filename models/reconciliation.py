"""
Hierarchical Forecast Reconciliation.

Ensures forecasts are coherent across the M5 hierarchy:
- Bottom-up: item-store → department → category → store → state → total
- Forecasts at aggregated levels should equal the sum of children

Uses a differentiable reconciliation layer inspired by MinT
(Minimum Trace optimal reconciliation, Wickramasuriya et al. 2019).

UPDATED: Simplified SimpleReconciliation to remove the double-softplus
chain that caused gradient instability.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional


class HierarchicalReconciliation(nn.Module):
    """
    Differentiable hierarchical forecast reconciliation.
    
    Given raw bottom-level forecasts, this module:
    1. Computes aggregated forecasts at all levels
    2. Applies a learned reconciliation matrix to ensure coherence
    3. Returns adjusted bottom-level forecasts
    
    The reconciliation matrix is parameterized as:
    ŷ_adjusted = S × P × ŷ_raw
    
    where:
    - S is the summing matrix (maps bottom to all levels)
    - P is a learned projection matrix
    """

    def __init__(
        self,
        num_bottom: int,
        hierarchy_groups: Optional[Dict[str, List[List[int]]]] = None,
        method: str = 'bottom_up',  # 'bottom_up', 'learned', 'ols'
    ):
        """
        Args:
            num_bottom: number of bottom-level series
            hierarchy_groups: dict of level_name → list of groups (each group = list of indices)
            method: reconciliation method
        """
        super().__init__()
        self.num_bottom = num_bottom
        self.method = method

        if method == 'learned' and hierarchy_groups is not None:
            # Build the summing matrix S
            num_aggregated = sum(len(groups) for groups in hierarchy_groups.values())
            total = num_bottom + num_aggregated

            S = torch.zeros(total, num_bottom)
            # Bottom level: identity
            S[:num_bottom, :num_bottom] = torch.eye(num_bottom)
            # Aggregated levels
            idx = num_bottom
            for level_name, groups in hierarchy_groups.items():
                for group in groups:
                    for member in group:
                        if member < num_bottom:
                            S[idx, member] = 1.0
                    idx += 1

            self.register_buffer('S', S)

            # Learnable reconciliation weights
            self.P = nn.Linear(total, num_bottom, bias=False)
            nn.init.eye_(self.P.weight[:num_bottom, :num_bottom])
        else:
            self.S = None
            self.P = None

    def forward(
        self,
        base_forecasts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply reconciliation to bottom-level forecasts.
        
        Args:
            base_forecasts: (N, horizon) raw bottom-level forecasts
            
        Returns:
            (N, horizon) reconciled forecasts
        """
        if self.method == 'bottom_up' or self.P is None:
            # Bottom-up: no adjustment needed for bottom level
            # Just ensure non-negativity
            return F.relu(base_forecasts)

        elif self.method == 'learned':
            # Stack bottom forecasts and aggregated
            # base_forecasts: (N, H)
            H = base_forecasts.size(1)

            # Compute aggregated forecasts for each horizon step
            reconciled = []
            for h in range(H):
                y_h = base_forecasts[:, h]  # (N,)

                # Compute all-level forecasts via summing matrix
                all_level = self.S @ y_h  # (total,)

                # Apply learned projection
                adjusted = self.P(all_level)  # (N,)

                reconciled.append(adjusted)

            return torch.stack(reconciled, dim=1)  # (N, H)

        return base_forecasts


class SimpleReconciliation(nn.Module):
    """
    Simple reconciliation layer that:
    1. Applies learned per-group scaling factors
    2. Enforces non-negative forecasts (sales can't be negative)
    3. Optionally clips extreme predictions
    
    UPDATED: Removed double-softplus chain that caused gradient instability.
    Now uses a single ReLU for non-negativity instead of softplus-on-softplus.
    """

    def __init__(self, num_groups: int = 7, max_ratio: float = 50.0):
        """
        Args:
            num_groups: number of department/category groups
            max_ratio: max ratio of prediction to historical mean
        """
        super().__init__()
        self.scale = nn.Parameter(torch.ones(num_groups))
        self.bias = nn.Parameter(torch.zeros(num_groups))
        self.max_ratio = max_ratio

    def forward(
        self,
        predictions: torch.Tensor,
        group_ids: Optional[torch.Tensor] = None,
        historical_mean: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = predictions
        
        # Per-group scaling
        if group_ids is not None:
            # ── FIX: Clamp group_ids to valid range to prevent index errors ──
            g = group_ids.clamp(0, len(self.scale) - 1)
            # Use softplus for scale (always positive), clamp to prevent explosion
            scale = torch.clamp(F.softplus(self.scale[g]), min=0.1, max=10.0).unsqueeze(1)  # (N, 1)
            bias = self.bias[g].unsqueeze(1)    # (N, 1)
            out = out * scale + bias
            
        # ── FIX: Simple ReLU instead of softplus ──
        # softplus(softplus(x)) has compounding gradient issues;
        # ReLU is piecewise linear → stable gradients
        out = F.relu(out)

        # Clip extreme predictions
        if historical_mean is not None:
            # ── FIX: Add 1.0 to prevent zero upper bound for items with 0 historical mean ──
            upper = (historical_mean.unsqueeze(1) + 1.0) * self.max_ratio
            out = torch.minimum(out, upper)

        return out
