import os
import torch
from torch_geometric.data import HeteroData
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def build_graph_data():
    print("📥 Loading generated entity and transaction data...")
    users = pd.read_csv("data/users.csv")
    transactions = pd.read_csv("data/transactions.csv")

    data = HeteroData()

    # --- 1. Node Features & Mapping ---
    # User features (age, credit_score)
    user_id_map = {uid: i for i, uid in enumerate(users['user_id'].unique())}
    scaler = StandardScaler()
    user_features = scaler.fit_transform(users[['age', 'credit_score']].values)
    data['user'].x = torch.tensor(user_features, dtype=torch.float)

    # Merchant features (synthetic embedding features)
    merchant_ids = transactions['merchant_id'].unique()
    merchant_id_map = {mid: i for i, mid in enumerate(merchant_ids)}
    data['merchant'].x = torch.randn((len(merchant_ids), 2), dtype=torch.float)

    # Transaction features (amount) & labels
    tx_features = scaler.fit_transform(transactions[['amount']].values)
    data['transaction'].x = torch.tensor(tx_features, dtype=torch.float)
    data['transaction'].y = torch.tensor(transactions['is_fraud'].values, dtype=torch.long)

    # --- 2. Construct Edges ---
    print("🕸️ Constructing heterogeneous graph topological edges...")
    
    # Edge Type 1: User -> performs -> Transaction
    user_indices = transactions['user_id'].map(user_id_map).values
    tx_indices = np.arange(len(transactions))
    
    edge_index_user_tx = torch.tensor([
        user_indices.tolist(),
        tx_indices.tolist()
    ], dtype=torch.long)
    data['user', 'performs', 'transaction'].edge_index = edge_index_user_tx

    # Edge Type 2: Transaction -> involves -> Merchant
    merchant_indices = transactions['merchant_id'].map(merchant_id_map).values
    edge_index_tx_merchant = torch.tensor([
        tx_indices.tolist(),
        merchant_indices.tolist()
    ], dtype=torch.long)
    data['transaction', 'involves', 'merchant'].edge_index = edge_index_tx_merchant

    print("✅ Heterogeneous Graph successfully constructed!")
    print(data)
    return data

if __name__ == "__main__":
    build_graph_data()