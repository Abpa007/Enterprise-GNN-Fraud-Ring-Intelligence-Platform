import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
import os
import sys

# Add project root to path so we can import from graph_engine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from graph_engine.builder import build_graph_data
from graph_engine.model import FraudGraphClassifier

def train_gnn():
    print("🚀 Initializing Heterogeneous GNN Training Pipeline...")
    data = build_graph_data()
    
    # Extract graph metadata needed for torch_geometric to_hetero wrapper
    metadata = data.metadata()
    
    # Instantiate our GNN Fraud Classifier
    model = FraudGraphClassifier(metadata, hidden_channels=64, out_channels=32)
    
    # Optimizer configuration
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    # Extract transaction labels
    labels = data['transaction'].y
    
    # Create train/test masks on transaction nodes (80% train, 20% test split)
    num_tx = labels.size(0)
    indices = torch.randperm(num_tx)
    train_size = int(0.8 * num_tx)
    
    train_mask = torch.zeros(num_tx, dtype=torch.bool)
    train_mask[indices[:train_size]] = True
    
    test_mask = torch.zeros(num_tx, dtype=torch.bool)
    test_mask[indices[train_size:]] = True

    print(f"📊 Dataset Split -> Training Nodes: {train_mask.sum().item():,} | Testing Nodes: {test_mask.sum().item():,}")
    print(f"🎯 Total Fraud Cases in Dataset: {(labels == 1).sum().item():,}")

    # Training loop across epochs
    model.train()
    for epoch in range(1, 16):  # 16 epochs for optimal convergence
        optimizer.zero_grad()
        
        # Forward pass: compute multi-hop message-passing embeddings
        out = model(data.x_dict, data.edge_index_dict)
        
        # Calculate Cross-Entropy Loss strictly on training transactions
        loss = F.cross_entropy(out[train_mask], labels[train_mask])
        
        loss.backward()
        optimizer.step()
        
        # Evaluate model performance on the test split
        model.eval()
        with torch.no_grad():
            test_out = model(data.x_dict, data.edge_index_dict)
            preds = test_out[test_mask].argmax(dim=-1)
            test_labels = labels[test_mask]
            
            acc = (preds == test_labels).float().mean().item()
            f1 = f1_score(test_labels.numpy(), preds.numpy(), zero_division=0)
            
        model.train()
        print(f"Epoch {epoch:02d} | Loss: {loss.item():.4f} | Test Accuracy: {acc:.4f} | Test F1-Score: {f1:.4f}")

    # Save trained model binary
    os.makedirs("models", exist_ok=True)
    model_path = "models/fraud_gnn.pt"
    torch.save(model.state_dict(), model_path)
    print(f"✅ GNN Model training complete! Binary successfully saved to `{model_path}`")

if __name__ == "__main__":
    train_gnn()