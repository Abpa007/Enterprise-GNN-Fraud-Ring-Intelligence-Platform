import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero

class GraphSAGEEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class FraudGraphClassifier(torch.nn.Module):
    def __init__(self, metadata, hidden_channels=64, out_channels=32):
        super().__init__()
        # 1. Base GraphSAGE encoder wrapper converted for Heterogeneous graphs
        base_encoder = GraphSAGEEncoder(hidden_channels, out_channels)
        self.gnn = to_hetero(base_encoder, metadata, aggr='sum')
        
        # 2. Executive classification head for transaction fraud detection
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(out_channels, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 2) # Binary classification: 0 (Legitimate), 1 (Fraud)
        )

    def forward(self, x_dict, edge_index_dict):
        # Pass features through the heterogeneous GNN message-passing layers
        node_embeddings = self.gnn(x_dict, edge_index_dict)
        
        # Extract embeddings specifically for transaction nodes to predict fraud
        tx_embeddings = node_embeddings['transaction']
        logits = self.classifier(tx_embeddings)
        return logits