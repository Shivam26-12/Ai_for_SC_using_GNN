"""
Sparse Temporal Graph Attention Network (GAT) with edge-type awareness.

Key innovations over standard GAT:
1. Edge-type embeddings for heterogeneous graphs (hierarchical, correlation, cross-store)
2. Multi-head attention with per-head specialization
3. Residual connections + Layer Normalization for deep stacking
4. Dropout on both attention coefficients and features
5. Memory-efficient: O(E) instead of O(N²)

UPDATED: 
- Dynamic device detection for autocast (works on both CUDA and CPU)
- BF16-safe: On A100 with BF16, the GAT can run in mixed precision
  (BF16 has same exponent range as FP32, no overflow risk)
- Forces FP32 for attention computation regardless of AMP dtype
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from typing import Optional


def _get_device_type(x: torch.Tensor) -> str:
    """Get the device type string for autocast."""
    if x.is_cuda:
        return 'cuda'
    return 'cpu'


class SparseGATLayer(nn.Module):
    """
    Single layer of the Sparse Graph Attention Network.
    
    For each edge (i → j):
    1. Compute attention coefficient α_{ij} using learned attention
    2. Modulate by edge type embedding
    3. Aggregate neighbor messages with attention weights
    
    Uses scatter operations for memory-efficient sparse computation.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
        num_edge_types: int = 3,
        dropout: float = 0.2,
        negative_slope: float = 0.2,
        concat_heads: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        self.head_dim = out_dim // num_heads if concat_heads else out_dim

        # ── Linear projections for Q, K, V ──
        self.W_q = nn.Linear(in_dim, num_heads * self.head_dim, bias=False)
        self.W_k = nn.Linear(in_dim, num_heads * self.head_dim, bias=False)
        self.W_v = nn.Linear(in_dim, num_heads * self.head_dim, bias=False)

        # ── Attention scoring ──
        self.attn_linear = nn.Parameter(torch.zeros(1, num_heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.attn_linear)

        # ── Edge type embeddings ──
        self.edge_type_embed = nn.Embedding(num_edge_types, num_heads * self.head_dim)

        # ── Output projection ──
        output_dim = num_heads * self.head_dim if concat_heads else self.head_dim
        self.out_proj = nn.Linear(output_dim, out_dim)

        # ── Regularization ──
        self.attn_dropout = nn.Dropout(dropout)
        self.feat_dropout = nn.Dropout(dropout)
        self.negative_slope = negative_slope

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (N, in_dim) node features
            edge_index: (2, E) edge indices [src, dst]
            edge_type: (E,) edge type for each edge
            
        Returns:
            (N, out_dim) updated node features
        """
        # ── CRITICAL: Disable AMP autocast for GAT attention computation ──
        # Attention scores and scatter_add must be in FP32 to avoid overflow.
        # On A100 with BF16, this is less critical but still safer.
        device_type = _get_device_type(x)
        with autocast(device_type=device_type, enabled=False):
            x = x.float()  # Ensure FP32 input
            
            N = x.size(0)
            H = self.num_heads
            D = self.head_dim

            src, dst = edge_index[0], edge_index[1]

            # ── Project to multi-head Q, K, V (all in FP32) ──
            q = self.W_q(x).view(N, H, D)  # (N, H, D)
            k = self.W_k(x).view(N, H, D)
            v = self.W_v(x).view(N, H, D)

            # ── Compute attention for each edge ──
            attn_l = self.attn_linear[:, :, :D]  # (1, H, D)
            attn_r = self.attn_linear[:, :, D:]  # (1, H, D)
            
            score_src = (q * attn_l).sum(dim=-1)  # (N, H)
            score_dst = (k * attn_r).sum(dim=-1)  # (N, H)
            
            attn_scores = score_src[src] + score_dst[dst]  # (E, H)
            attn_scores = F.leaky_relu(attn_scores, self.negative_slope)

            # ── Modulate by edge type ──
            if edge_type is not None:
                q_src = q[src]
                edge_embed = self.edge_type_embed(edge_type).view(-1, H, D)  # (E, H, D)
                type_score = (q_src * edge_embed).sum(dim=-1)  # (E, H)
                attn_scores = attn_scores + type_score

            # ── Softmax over neighbors (per-node normalization) ──
            attn_scores = self._sparse_softmax(attn_scores, dst, N)  # (E, H)
            attn_scores = self.attn_dropout(attn_scores)

            # ── Message passing: weighted sum of neighbor values ──
            v_src = v[src]  # (E, H, D)
            messages = attn_scores.unsqueeze(-1) * v_src  # (E, H, D)

            # Scatter/aggregate messages to destination nodes
            out = torch.zeros(N, H, D, device=x.device, dtype=torch.float32)
            out.scatter_add_(0, dst.unsqueeze(-1).unsqueeze(-1).expand_as(messages), messages)

            # ── Clamp output to prevent extreme activations ──
            out = torch.clamp(out, min=-50.0, max=50.0)

            # ── Reshape and project ──
            if self.concat_heads:
                out = out.reshape(N, H * D)  # (N, H*D)
            else:
                out = out.mean(dim=1)  # (N, D)

            out = self.out_proj(out)  # (N, out_dim)
            out = self.feat_dropout(out)

            return out

    def _sparse_softmax(
        self, 
        scores: torch.Tensor, 
        index: torch.Tensor, 
        num_nodes: int
    ) -> torch.Tensor:
        """
        Compute softmax over edges grouped by destination node.
        Gradient-friendly: uses detached max for numerical stability.
        """
        # Numerical stability: subtract max per node (detached, no grad needed)
        with torch.no_grad():
            max_scores = torch.full(
                (num_nodes, scores.size(1)), float('-inf'), 
                device=scores.device, dtype=scores.dtype
            )
            max_scores.scatter_reduce_(
                0,
                index.unsqueeze(-1).expand_as(scores),
                scores.detach(),
                reduce='amax',
                include_self=False
            )
            # Replace -inf with 0 for island nodes (no incoming edges)
            max_scores = max_scores.clamp(min=-100.0)

        scores = scores - max_scores[index]

        # Exponentiate (safe since we subtracted max)
        exp_scores = torch.exp(scores)

        # Sum per node
        sum_exp = torch.zeros(num_nodes, scores.size(1), 
                              device=scores.device, dtype=scores.dtype)
        sum_exp.scatter_add_(0, index.unsqueeze(-1).expand_as(exp_scores), exp_scores)

        # Use larger epsilon to prevent near-zero division
        denom = sum_exp[index] + 1e-6
        result = exp_scores / denom
        
        # Clamp attention weights
        return result.clamp(max=1.0)


class SparseTemporalGAT(nn.Module):
    """
    Multi-layer Sparse Temporal GAT with residual connections.
    
    Architecture per layer:
    x → GAT → LayerNorm → Residual Add → GELU → Dropout
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        num_edge_types: int = 3,
        dropout: float = 0.1,
        residual: bool = True,
        layer_norm: bool = True,
    ):
        super().__init__()
        self.residual = residual
        self.num_layers = num_layers

        # Input projection (if dimensions don't match)
        self.input_proj = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()

        # GAT layers
        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()

        for i in range(num_layers):
            self.gat_layers.append(
                SparseGATLayer(
                    in_dim=hidden_dim,
                    out_dim=hidden_dim,
                    num_heads=num_heads,
                    num_edge_types=num_edge_types,
                    dropout=dropout,
                )
            )
            if layer_norm:
                self.norms.append(nn.LayerNorm(hidden_dim))
            else:
                self.norms.append(nn.Identity())

            # Feed-forward network after each GAT layer
            self.ffn_layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Dropout(dropout),
            ))

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (N, in_dim) node features
            edge_index: (2, E) edge indices
            edge_type: (E,) edge types
            
        Returns:
            (N, out_dim) updated node features
        """
        h = self.input_proj(x)

        for i in range(self.num_layers):
            # GAT message passing
            h_new = self.gat_layers[i](h, edge_index, edge_type)

            # Residual + LayerNorm
            if self.residual:
                h_new = h_new + h
            h_new = self.norms[i](h_new)

            # FFN
            h_ffn = self.ffn_layers[i](h_new)
            if self.residual:
                h_new = h_new + h_ffn
            else:
                h_new = h_ffn

            # Guard after each layer
            h_new = torch.nan_to_num(h_new, nan=0.0, posinf=50.0, neginf=-50.0)
            h = h_new

        return self.output_proj(h)
