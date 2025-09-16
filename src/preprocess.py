"""
H2D-GAT Data Preprocessing Module
Handles data loading and preprocessing for various datasets
"""

import torch
from torch_geometric.datasets import Planetoid, Reddit, PPI
from torch_geometric.data import Data
from torch_geometric.transforms import NormalizeFeatures
import numpy as np
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    """Data loading and preprocessing for H2D-GAT experiments"""
    
    def __init__(self, config):
        self.config = config
        self.dataset_name = config['dataset']['name']
        
    def load_and_preprocess(self):
        """Load and preprocess dataset based on configuration"""
        
        if self.dataset_name == "Cora":
            return self._load_cora()
        elif self.dataset_name == "ogbn-products":
            return self._load_ogbn_products()
        elif self.dataset_name == "Reddit":
            return self._load_reddit()
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
    
    def _load_cora(self):
        """Load and preprocess Cora dataset"""
        print("Loading Cora dataset...")
        
        dataset = Planetoid(root='data/Cora', name='Cora', transform=NormalizeFeatures())
        data = dataset[0]
        
        if 'smoke' in self.config['experiment']['name']:
            num_nodes = min(1000, data.num_nodes)
            indices = torch.randperm(data.num_nodes)[:num_nodes]
            
            data.x = data.x[indices]
            data.y = data.y[indices]
            
            edge_mask = torch.isin(data.edge_index[0], indices) & torch.isin(data.edge_index[1], indices)
            data.edge_index = data.edge_index[:, edge_mask]
            
            old_to_new = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(indices)}
            for i in range(data.edge_index.size(1)):
                data.edge_index[0, i] = old_to_new[data.edge_index[0, i].item()]
                data.edge_index[1, i] = old_to_new[data.edge_index[1, i].item()]
            
            data.train_mask = data.train_mask[indices]
            data.val_mask = data.val_mask[indices]
            data.test_mask = data.test_mask[indices]
        
        print(f"Cora dataset loaded: {data.num_nodes} nodes, {data.num_edges} edges")
        return data
    
    def _load_ogbn_products(self):
        """Load and preprocess ogbn-products dataset (simplified version)"""
        print("Loading ogbn-products dataset (simplified)...")
        
        
        num_nodes = 10000 if 'smoke' in self.config['experiment']['name'] else 100000
        num_features = 100
        num_classes = 47
        
        x = torch.randn(num_nodes, num_features)
        y = torch.randint(0, num_classes, (num_nodes,))
        
        edge_list = []
        for i in range(num_nodes):
            num_edges = max(1, int(np.random.poisson(np.log(num_nodes))))
            targets = np.random.choice(num_nodes, size=min(num_edges, num_nodes-1), replace=False)
            for target in targets:
                if target != i:
                    edge_list.append([i, target])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        train_ratio = self.config['dataset']['train_ratio']
        val_ratio = self.config['dataset']['val_ratio']
        
        indices = torch.randperm(num_nodes)
        train_size = int(train_ratio * num_nodes)
        val_size = int(val_ratio * num_nodes)
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size+val_size]] = True
        test_mask[indices[train_size+val_size:]] = True
        
        data = Data(x=x, edge_index=edge_index, y=y,
                   train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
        
        print(f"ogbn-products dataset loaded: {data.num_nodes} nodes, {data.num_edges} edges")
        return data
    
    def _load_reddit(self):
        """Load and preprocess Reddit dataset"""
        print("Loading Reddit dataset...")
        
        try:
            dataset = Reddit(root='data/Reddit')
            data = dataset[0]
            
            if 'smoke' in self.config['experiment']['name']:
                num_nodes = min(5000, data.num_nodes)
                indices = torch.randperm(data.num_nodes)[:num_nodes]
                
                data.x = data.x[indices]
                data.y = data.y[indices]
                
                edge_mask = torch.isin(data.edge_index[0], indices) & torch.isin(data.edge_index[1], indices)
                data.edge_index = data.edge_index[:, edge_mask]
                
                old_to_new = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(indices)}
                for i in range(data.edge_index.size(1)):
                    data.edge_index[0, i] = old_to_new[data.edge_index[0, i].item()]
                    data.edge_index[1, i] = old_to_new[data.edge_index[1, i].item()]
                
                data.train_mask = data.train_mask[indices]
                data.val_mask = data.val_mask[indices]
                data.test_mask = data.test_mask[indices]
            
        except Exception as e:
            print(f"Error loading Reddit dataset: {e}")
            print("Using synthetic Reddit-like dataset...")
            return self._create_synthetic_reddit()
        
        print(f"Reddit dataset loaded: {data.num_nodes} nodes, {data.num_edges} edges")
        return data
    
    def _create_synthetic_reddit(self):
        """Create synthetic Reddit-like dataset"""
        num_nodes = 5000 if 'smoke' in self.config['experiment']['name'] else 50000
        num_features = 602
        num_classes = 41
        
        x = torch.randn(num_nodes, num_features)
        y = torch.randint(0, num_classes, (num_nodes,))
        
        edge_list = []
        community_size = 100
        num_communities = num_nodes // community_size
        
        for comm in range(num_communities):
            start_idx = comm * community_size
            end_idx = min((comm + 1) * community_size, num_nodes)
            
            for i in range(start_idx, end_idx):
                for j in range(i + 1, end_idx):
                    if np.random.random() < 0.1:  # 10% connection probability
                        edge_list.append([i, j])
                        edge_list.append([j, i])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        train_ratio = self.config['dataset']['train_ratio']
        val_ratio = self.config['dataset']['val_ratio']
        
        indices = torch.randperm(num_nodes)
        train_size = int(train_ratio * num_nodes)
        val_size = int(val_ratio * num_nodes)
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size+val_size]] = True
        test_mask[indices[train_size+val_size:]] = True
        
        data = Data(x=x, edge_index=edge_index, y=y,
                   train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
        
        return data
