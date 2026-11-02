import os
import torch
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graph_engine.builder import build_graph_data
from graph_engine.model import FraudGraphClassifier

def init_vector_database():
    print("🚀 Initializing Qdrant Vector Store for Fraud Pattern Matching...")
    
    # 1. Initialize local Qdrant client (in-memory or persistent local storage)
    os.makedirs("models/qdrant_storage", exist_ok=True)
    client = QdrantClient(path="models/qdrant_storage")
    
    collection_name = "fraud_transactions"
    
    # Recreate collection if it already exists
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
        
    # Vector dimension from our GNN output layer (out_channels = 32)
    vector_size = 32
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )
    print(f"📦 Qdrant collection `{collection_name}` created successfully.")

    # 2. Load Graph Data & Trained GNN Model to extract embeddings
    data = build_graph_data()
    metadata = data.metadata()
    
    model = FraudGraphClassifier(metadata)
    model_path = "models/fraud_gnn.pt"
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("🧠 Loaded trained GNN model weights successfully.")
    else:
        print("⚠️ Warning: Trained GNN model not found. Using untrained initialization for demo.")

    model.eval()
    with torch.no_grad():
        # Get node embeddings from GNN
        node_embeddings = model.gnn(data.x_dict, data.edge_index_dict)
        tx_embeddings = node_embeddings['transaction'].numpy()

    # 3. Upsert transaction vectors into Qdrant
    print("⬆️ Indexing transaction vectors into Qdrant vector database...")
    labels = data['transaction'].y.numpy()
    
    points = []
    # Index a sample of transactions (e.g., first 5,000 for fast indexing performance)
    limit = min(5000, len(tx_embeddings))
    for i in range(limit):
        points.append(
            PointStruct(
                id=i,
                vector=tx_embeddings[i].tolist(),
                payload={
                    "transaction_index": i,
                    "is_fraud": int(labels[i])
                }
            )
        )
    
    client.upsert(
        collection_name=collection_name,
        points=points
    )
    print(f"✅ Successfully indexed {limit:,} transaction vectors into Qdrant!")
    return client

if __name__ == "__main__":
    init_vector_database()