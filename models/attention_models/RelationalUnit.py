import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class RelationLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, num_heads=8, hidden_size=1024, dropout_rate=0.5):
        super(RelationLayer, self).__init__(aggr='add')

        assert in_channels % num_heads == 0, \
            f"in_channels ({in_channels}) must be divisible by num_heads ({num_heads})"

        self.num_heads = num_heads
        self.head_dim  = in_channels // num_heads
        self.scale     = self.head_dim ** 0.5

        # Attention projections (all operate in in_channels space)
        self.query = nn.Linear(in_channels, in_channels)
        self.key   = nn.Linear(in_channels, in_channels)
        self.value = nn.Linear(2 * in_channels, in_channels)

        # Post-attention residual + norm (still in in_channels space)
        self.ln1 = nn.LayerNorm(in_channels)
        self.dr1 = nn.Dropout(dropout_rate)

        # FFN: projects from in_channels → out_channels
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, out_channels),
        )

        # Projection for the residual so dimensions match after FFN
        self.residual_proj = (
            nn.Linear(in_channels, out_channels, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.ln2 = nn.LayerNorm(out_channels)
        self.dr2 = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index):
        """
        x          : (B, N, in_channels)  — batched node features
        edge_index : (2, E)               — shared graph topology
        """
        # --- Multi-head self-attention via message passing ---
        x_att = self.propagate(edge_index, x=x)   # (B, N, in_channels)

        # Residual + LayerNorm  (both tensors are in_channels wide)
        x_att = self.ln1(x + self.dr1(x_att))     # (B, N, in_channels)

        # FFN + residual (project x_att to out_channels for the skip)
        x_ffn = self.ffn(x_att)                   # (B, N, out_channels)
        out   = self.ln2(self.residual_proj(x_att) + self.dr2(x_ffn))  # (B, N, out_channels)

        return out

    def message(self, x_i, x_j, index, ptr, size_i):
        """
        x_i, x_j : (B, E, in_channels)
        Returns   : (B, E, in_channels)
        """
        B, E, _ = x_i.shape

        # Reshape for multi-head attention
        Q = self.query(x_i).view(B, E, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, E, D)
        K = self.key(x_j).view(B, E, self.num_heads, self.head_dim).transpose(1, 2)    # (B, H, E, D)
        V = self.value(torch.cat([x_i, x_j], dim=-1)) \
                .view(B, E, self.num_heads, self.head_dim).transpose(1, 2)              # (B, H, E, D)

        # Scaled dot-product scores
        e_ij = (Q * K).sum(dim=-1) / self.scale  # (B, H, E)

        # Softmax normalised over destination nodes
        a_ij = softmax(e_ij, index, ptr, num_nodes=size_i, dim=-1)  # (B, H, E)

        # Weighted values → merge heads → (B, E, in_channels)
        out = (a_ij.unsqueeze(-1) * V).transpose(1, 2).contiguous().view(B, E, self.num_heads * self.head_dim)
        return out

    def update(self, aggr_out):
        return aggr_out