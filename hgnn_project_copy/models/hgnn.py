import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
import math

class PoincareBall:
    """Poincaré ball model for hyperbolic geometry with learnable curvature"""
    def __init__(self, c=1.0):
        self.c = nn.Parameter(torch.tensor(c, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-5

    def mobius_add(self, x, y):
        x2 = torch.sum(x * x, dim=-1, keepdim=True)
        y2 = torch.sum(y * y, dim=-1, keepdim=True)
        xy = torch.sum(x * y, dim=-1, keepdim=True)

        num = (1 + 2 * self.c * xy + self.c * y2) * x + (1 - self.c * x2) * y
        denom = 1 + 2 * self.c * xy + self.c * x2 * y2

        return num / (denom + self.eps)

    def exp_map(self, v, x):
        v_norm = torch.norm(v, dim=-1, keepdim=True)
        v_norm = torch.clamp(v_norm, min=self.eps)

        sqrt_c = torch.sqrt(self.c)
        lambda_x = 2 / (1 - self.c * torch.sum(x * x, dim=-1, keepdim=True))

        direction = v / v_norm
        factor = torch.tanh(sqrt_c * lambda_x * v_norm / 2) / (sqrt_c * v_norm)

        return self.mobius_add(x, factor * v)

    def log_map(self, y, x):
        diff = self.mobius_add(-x, y)
        diff_norm = torch.norm(diff, dim=-1, keepdim=True)
        diff_norm = torch.clamp(diff_norm, min=self.eps)

        sqrt_c = torch.sqrt(self.c)
        lambda_x = 2 / (1 - self.c * torch.sum(x * x, dim=-1, keepdim=True))

        factor = 2 / (sqrt_c * lambda_x) * torch.atanh(sqrt_c * diff_norm) / diff_norm

        return factor * diff

class HyperbolicLinear(nn.Module):
    def __init__(self, in_features, out_features, manifold, bias=True):
        super(HyperbolicLinear, self).__init__()
        self.manifold = manifold
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        x_tangent = self.manifold.log_map(x, torch.zeros_like(x))
        out = F.linear(x_tangent, self.weight, self.bias)
        return self.manifold.exp_map(out, torch.zeros_like(out))

class HyperbolicGraphConv(MessagePassing):
    def __init__(self, in_channels, out_channels, manifold, aggr='mean'):
        super(HyperbolicGraphConv, self).__init__(aggr=aggr)
        self.lin_self = HyperbolicLinear(in_channels, out_channels, manifold)
        self.lin_neigh = HyperbolicLinear(in_channels, out_channels, manifold)
        self.manifold = manifold

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        out = self.propagate(edge_index, x=x)
        x_self = self.lin_self(x)
        return self.manifold.mobius_add(x_self, out)

    def message(self, x_j):
        return self.lin_neigh(x_j)

class HGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2, c=1.0):
        super(HGNN, self).__init__()
        self.manifold = PoincareBall(c=c)

        self.input_proj = HyperbolicLinear(input_dim, hidden_dim, self.manifold)

        self.convs = nn.ModuleList([
            HyperbolicGraphConv(hidden_dim, hidden_dim, self.manifold)
            for _ in range(num_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_classes),
            #nn.Tanh(),
            #nn.Linear(hidden_dim, num_classes)
        )

        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index, batch=None):
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
        x = self.input_proj(x)

        for conv in self.convs:
            x_new = conv(x, edge_index)
            x_tangent = self.manifold.log_map(x_new, torch.zeros_like(x_new))
            x_tangent = torch.tanh(x_tangent)
            x_tangent = self.dropout(x_tangent)
            x = self.manifold.exp_map(x_tangent, torch.zeros_like(x_tangent))

        x = self.manifold.log_map(x, torch.zeros_like(x))
        x = self.classifier(x)
        return x
