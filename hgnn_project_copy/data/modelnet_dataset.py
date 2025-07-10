import os
import pickle
import torch
from torch_geometric.data import Data, Dataset
import numpy as np

class ModelNetGraphDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        self.root = root
        super().__init__(root, transform, pre_transform)
        self.graph_files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith('.pkl')]
        
        # Get all unique labels to create proper label mapping
        self.labels = []
        for file_path in self.graph_files:
            with open(file_path, 'rb') as f:
                graph = pickle.load(f)
                self.labels.append(graph['label'])
        
        # Create label mapping to ensure labels are 0-indexed and contiguous
        unique_labels = sorted(list(set(self.labels)))
        self.label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        self._num_classes = len(unique_labels)
        
        print(f"Found {len(self.graph_files)} graphs with {self.num_classes} classes")
        print(f"Label mapping: {self.label_mapping}")

    def len(self):
        return len(self.graph_files)

    def get(self, idx):
        with open(self.graph_files[idx], 'rb') as f:
            graph = pickle.load(f)

        # Convert features to tensor
        x = torch.tensor(graph['features'], dtype=torch.float)
        
        # Map label using the label mapping
        original_label = graph['label']
        mapped_label = self.label_mapping[original_label]
        y = torch.tensor([mapped_label], dtype=torch.long)  # ✅ FIXED

        # Convert adjacency matrix to edge_index
        edge_index = self.adj_to_edge_index(graph['adj'])
        
        # Create PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index, y=y)
        
        return data


    def adj_to_edge_index(self, adj):
        """Convert adjacency matrix to edge_index format"""
        # Find non-zero entries (edges)
        row, col = np.where(adj > 0)
        edge_index = torch.from_numpy(np.stack([row, col], axis=0)).long()
        return edge_index
    
    @property
    def num_features(self):
        """Return number of node features"""
        if len(self.graph_files) > 0:
            with open(self.graph_files[0], 'rb') as f:
                graph = pickle.load(f)
            return graph['features'].shape[1]
        return 3  # Default for 3D coordinates
    
    @property  
    def num_node_features(self):
        return self._num_features