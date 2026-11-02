import os
import pandas as pd
import numpy as np
from faker import Faker

fake = Faker()
np.random.seed(42)

def generate_graph_dataset(num_users=5000, num_transactions=50000):
    os.makedirs("data", exist_ok=True)
    print("🌐 Generating multi-entity financial graph telemetry...")

    # 1. Generate Entities
    users = pd.DataFrame({
        'user_id': [f"U_{i:05d}" for i in range(num_users)],
        'age': np.random.randint(18, 70, size=num_users),
        'credit_score': np.random.randint(300, 850, size=num_users)
    })

    devices = [f"DEV_{i:04d}" for i in range(int(num_users * 0.4))]
    ips = [f"192.168.1.{i%255}" for i in range(int(num_users * 0.3))]
    merchants = [f"MCH_{i:03d}" for i in range(100)]

    # Map users to shared devices/IPs (injecting fraud ring clusters)
    user_devices = np.random.choice(devices, size=num_users)
    user_ips = np.random.choice(ips, size=num_users)

    users['device_id'] = user_devices
    users['ip_address'] = user_ips

    # 2. Generate Transactions connecting Users to Merchants
    print("💳 Simulating transactions and edge interactions...")
    tx_user_idx = np.random.choice(num_users, size=num_transactions)
    tx_merchants = np.random.choice(merchants, size=num_transactions)
    
    amounts = np.random.exponential(scale=150.0, size=num_transactions) + 10.0
    timestamps = pd.date_range(start="2026-01-01", periods=num_transactions, freq="s")

    transactions = pd.DataFrame({
        'transaction_id': [f"TX_{i:08d}" for i in range(num_transactions)],
        'user_id': users['user_id'].iloc[tx_user_idx].values,
        'merchant_id': tx_merchants,
        'amount': np.round(amounts, 2),
        'timestamp': timestamps
    })

    # Inject Fraud Labels (Coordinate fraud rings: certain devices/IPs have high fraud probability)
    # Fraud ring clusters share specific rogue devices
    rogue_devices = devices[:10]
    is_rogue_device = users.set_index('user_id')['device_id'].isin(rogue_devices).reindex(transactions['user_id']).values
    
    fraud_probs = np.where(is_rogue_device, 0.45, 0.01) # 45% fraud rate for rogue ring, 1% baseline
    transactions['is_fraud'] = np.random.binomial(1, fraud_probs)

    # Save to CSV
    users.to_csv("data/users.csv", index=False)
    transactions.to_csv("data/transactions.csv", index=False)
    
    print(f"✅ Graph entities saved! Total Transactions: {num_transactions:,} | Fraud Cases: {transactions['is_fraud'].sum():,}")

if __name__ == "__main__":
    generate_graph_dataset()