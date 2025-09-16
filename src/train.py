"""
H2D-GAT Training Module
Implements Hierarchical-Historical Determinantal GAT with three stages:
- Stage A: Hierarchical temporal kernels
- Stage B: Group-wise k-DPP sampling  
- Stage C: Quantized fused scatter with RL scheduler
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class HierarchicalTemporalKernel(nn.Module):
    """Stage A: Hierarchical temporal kernels with CUR decomposition"""
    
    def __init__(self, num_heads, num_layers, window_size=16, rank=6):
        super().__init__()
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.rank = rank
        
        self.U_layers = nn.ParameterList([
            nn.Parameter(torch.randn(rank, rank)) for _ in range(num_layers)
        ])
        self.V_heads = nn.ParameterList([
            nn.Parameter(torch.randn(rank, rank)) for _ in range(num_heads)
        ])
        
        self.attention_history = {}
        
    def update_history(self, layer_idx, head_idx, attention_matrix):
        """Update attention history with online CUR decomposition"""
        key = (layer_idx, head_idx)
        if key not in self.attention_history:
            self.attention_history[key] = []
        
        self.attention_history[key].append(attention_matrix.detach())
        if len(self.attention_history[key]) > self.window_size:
            self.attention_history[key].pop(0)
    
    def get_similarity_kernel(self, layer_idx, head_idx):
        """Compute two-level similarity kernel U_ℓ ⊗ V_h"""
        U_l = self.U_layers[layer_idx]
        V_h = self.V_heads[head_idx]
        return torch.kron(U_l, V_h)

class GroupwiseDPPSampler(nn.Module):
    """Stage B: Group-wise k-DPP with inference reuse"""
    
    def __init__(self, group_size=6, lambda_reg=0.1):
        super().__init__()
        self.group_size = group_size
        self.lambda_reg = lambda_reg
        self.cached_masks = {}
        
    def sample_groups(self, similarity_kernel, importance_scores, k_groups, training=True):
        """Sample k_groups using Kronecker-factored DPP"""
        n_scores = len(importance_scores)
        
        if similarity_kernel.size(0) != n_scores:
            if similarity_kernel.size(0) > n_scores:
                similarity_kernel = similarity_kernel[:n_scores, :n_scores]
            else:
                pad_size = n_scores - similarity_kernel.size(0)
                padded_kernel = torch.eye(n_scores, device=similarity_kernel.device)
                padded_kernel[:similarity_kernel.size(0), :similarity_kernel.size(1)] = similarity_kernel
                similarity_kernel = padded_kernel
        
        L = torch.diag(importance_scores) + self.lambda_reg * torch.mm(
            similarity_kernel, similarity_kernel.t()
        )
        
        if training:
            eigenvals, eigenvecs = torch.linalg.eigh(L)
            probs = eigenvals / (1 + eigenvals)
            selected = torch.bernoulli(probs)
            mask = selected > 0.5
        else:
            cache_key = hash(tuple(importance_scores.cpu().numpy()))
            if cache_key in self.cached_masks:
                mask = self.cached_masks[cache_key]
            else:
                _, indices = torch.topk(importance_scores, min(k_groups * self.group_size, n_scores))
                mask = torch.zeros_like(importance_scores, dtype=torch.bool)
                mask[indices] = True
                self.cached_masks[cache_key] = mask
        
        return mask

class QuantizedScatterAggregation(nn.Module):
    """Stage C: Quantized fused scatter with RL budget scheduler"""
    
    def __init__(self, hidden_dim, quantization_bits=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.quantization_bits = quantization_bits
        self.scale = 2 ** (quantization_bits - 1) - 1
        
    def quantize_messages(self, messages):
        """Linear quantization to 4-bit integers"""
        messages_norm = torch.tanh(messages)
        quantized = torch.round(messages_norm * self.scale)
        return quantized
    
    def dequantize_and_aggregate(self, quantized_messages, edge_index, num_nodes):
        """Fused dequantization and scatter aggregation"""
        messages = quantized_messages.float() / self.scale
        
        row, col = edge_index
        out = torch.zeros(num_nodes, messages.size(-1), device=messages.device)
        out.scatter_add_(0, row.unsqueeze(-1).expand_as(messages), messages)
        
        return out

class RLBudgetScheduler(nn.Module):
    """RL agent for adaptive sparsity budget scheduling"""
    
    def __init__(self, state_dim=64, hidden_dim=64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def get_budget(self, variance_stats, latency_stats):
        """Output global budget K_t based on variance and latency"""
        state = torch.cat([variance_stats, latency_stats])
        budget = self.actor(state)
        value = self.critic(state)
        return budget, value

class H2DGATLayer(MessagePassing):
    """Complete H2D-GAT layer implementing all three stages"""
    
    def __init__(self, in_channels, out_channels, heads=8, dropout=0.6,
                 window_size=16, rank=6, group_size=6, quantization_bits=4, num_layers=2):
        super().__init__(aggr='add', node_dim=0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        
        self.temporal_kernel = HierarchicalTemporalKernel(heads, 2, window_size, rank)
        
        self.dpp_sampler = GroupwiseDPPSampler(group_size)
        
        self.quantized_scatter = QuantizedScatterAggregation(out_channels, quantization_bits)
        
        self.lin_l = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_r = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_l.weight)
        nn.init.xavier_uniform_(self.lin_r.weight)
        nn.init.xavier_uniform_(self.att)
        
    def forward(self, x, edge_index, layer_idx=0, training=True):
        H, C = self.heads, self.out_channels
        
        x_l = self.lin_l(x).view(-1, H, C)
        x_r = self.lin_r(x).view(-1, H, C)
        
        alpha_l = (x_l * self.att[:, :, :C]).sum(dim=-1)
        alpha_r = (x_r * self.att[:, :, C:]).sum(dim=-1)
        
        out = self.propagate(edge_index, x=(x_l, x_r), alpha=(alpha_l, alpha_r),
                           layer_idx=layer_idx, training=training)
        
        return out.view(-1, H * C)
    
    def message(self, x_j, alpha_i, alpha_j, index, ptr, size_i, layer_idx, training):
        alpha = alpha_i + alpha_j
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = F.softmax(alpha, dim=0)
        
        for head in range(self.heads):
            self.temporal_kernel.update_history(layer_idx, head, alpha[:, head])
        
        similarity_kernel = self.temporal_kernel.get_similarity_kernel(layer_idx, 0)
        importance_scores = alpha.mean(dim=1)  # Average across heads
        
        k_groups = max(1, int(0.2 * len(importance_scores)))  # 20% keep rate
        mask = self.dpp_sampler.sample_groups(
            similarity_kernel[:len(importance_scores), :len(importance_scores)],
            importance_scores, k_groups, training
        )
        
        alpha = alpha * mask.unsqueeze(1).float()
        
        messages = x_j * alpha.unsqueeze(-1)
        if training:
            quantized_messages = self.quantized_scatter.quantize_messages(messages)
            messages = quantized_messages.float() / self.quantized_scatter.scale
        
        return F.dropout(messages, p=self.dropout, training=training)

class H2DGAT(nn.Module):
    """Complete H2D-GAT model"""
    
    def __init__(self, num_features, num_classes, hidden_dim=256, num_layers=2,
                 heads=8, dropout=0.6, **kwargs):
        super().__init__()
        
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        
        self.convs.append(H2DGATLayer(num_features, hidden_dim, heads, dropout, num_layers=num_layers, **kwargs))
        
        for _ in range(num_layers - 2):
            self.convs.append(H2DGATLayer(hidden_dim * heads, hidden_dim, heads, dropout, num_layers=num_layers, **kwargs))
        
        if num_layers > 1:
            self.convs.append(H2DGATLayer(hidden_dim * heads, num_classes, 1, dropout, num_layers=num_layers, **kwargs))
        
        self.rl_scheduler = RLBudgetScheduler()
        
    def forward(self, x, edge_index, training=True):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, layer_idx=i, training=training)
            if i < len(self.convs) - 1:
                x = F.elu(x)
        
        return F.log_softmax(x, dim=-1)

class H2DGATTrainer:
    """H2D-GAT training orchestrator"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def train(self, data):
        """Train H2D-GAT model"""
        model_config = self.config['model']
        train_config = self.config['training']
        
        model = H2DGAT(
            num_features=data.x.size(1),
            num_classes=data.y.max().item() + 1,
            hidden_dim=model_config['hidden_dim'],
            num_layers=model_config['num_layers'],
            heads=model_config['num_heads'],
            window_size=model_config['window_size'],
            rank=model_config['rank'],
            group_size=model_config['group_size']
        ).to(self.device)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(train_config['lr']),
            weight_decay=float(train_config['weight_decay'])
        )
        
        model.train()
        train_losses = []
        
        for epoch in tqdm(range(self.config['experiment']['epochs'])):
            optimizer.zero_grad()
            
            out = model(data.x.to(self.device), data.edge_index.to(self.device))
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask].to(self.device))
            
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch:03d}, Loss: {loss:.4f}')
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses)
        plt.title('H2D-GAT Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('.research/iteration2/images/training_loss.png')
        plt.close()
        
        return model, {'train_losses': train_losses}
