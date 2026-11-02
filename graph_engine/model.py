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
        base_encoder = GraphSAGEEncoder(hidden_channels, out_channels)
        # Convert base encoder to handle all heterogeneous node and edge types
        self.gnn = to_hetero(base_encoder, metadata, aggr='sum')
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(out_channels, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 2)
        )

    def forward(self, x_dict, edge_index_dict):
        node_embeddings = self.gnn(x_dict, edge_index_dict)
        tx_embeddings = node_embeddings['transaction']
        logits = self.classifier(tx_embeddings)
        return logits