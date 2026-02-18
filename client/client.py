import flwr as fl
import pandas as pd
import numpy as np
import sys
sys.dont_write_bytecode = True
import os
TF_ENABLE_ONEDNN_OPTS=0
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input

# Add the root directory to path so we can find local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from .trust_rules import ResearchTrustLayer # Note the dot (.) before trust_rules
except ImportError:
    from trust_rules import ResearchTrustLayer
from utils.hash_utils import hash_model_weights
from blockchain.blockchain import Blockchain

def build_model(input_dim):
    """Simple neural network for binary classification."""
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, X, y, trust_weights, client_id):
        self.model = model
        self.X = X
        self.y = y
        self.trust_weights = trust_weights
        self.client_id = client_id
        # Initialize blockchain access for verification
        self.blockchain_verifier = Blockchain()

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        """
        Modified fit method with Blockchain Verification.
        Ensures the global model has not been tampered with.
        """
        # 1. Integrity Audit
        received_hash = hash_model_weights(parameters)
        
        # Refresh blockchain data from disk
        self.blockchain_verifier = Blockchain()
        latest_block = self.blockchain_verifier.chain[-1]
        
        print(f"\n[Client {self.client_id}] 🛡️ Security Audit - Round {latest_block.index + 1}")
        
        # Verification Logic
        if latest_block.index == 0:
            print(f"[Client {self.client_id}] 📜 Accepting Initial Global Model (Genesis).")
        else:
            if received_hash == latest_block.model_hash:
                print(f"✅ VERIFIED: Global model matches Blockchain.")
            else:
                print(f"❌ SECURITY ALERT: Model Hash Mismatch!")
                # This makes the rejection "active" and prevents training
                raise SystemExit("Terminating: Untrusted model detected. Audit failure.")

        # 2. Local Training
        self.model.set_weights(parameters)
        history = self.model.fit(
            self.X, self.y, 
            sample_weight=self.trust_weights, 
            epochs=1,
            verbose=1
        )
        
        print(f"[Client {self.client_id}] 📈 Local training complete on vetted data.")
        return self.model.get_weights(), len(self.X), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        # CHANGE 'y' TO 'self.y' BELOW:
        loss, acc = self.model.evaluate(self.X, self.y, verbose=0)
        return loss, len(self.X), {"accuracy": acc}

def start_client():
    if len(sys.argv) < 3:
        print("Usage: python client.py <csv_path> <client_id>")
        return

    csv_path = sys.argv[1]
    client_id = sys.argv[2]

    # --- Step 1: Data Pre-processing via Neuro-Symbolic Layer ---
    print(f"\n[Client {client_id}] 🔍 Filtering data through NeuroTrust Layer...")
    df = pd.read_csv(csv_path)
    trust_layer = ResearchTrustLayer()
    
    valid_df, elim_df, trust_weights = trust_layer.filter_data(df)
    
    # Save Audit Files
    valid_df.to_csv(f"client_{client_id}_used_data.csv", index=False)
    elim_df.to_csv(f"client_{client_id}_eliminated_data.csv", index=False)
    
    print(f"[Client {client_id}] ✅ Filtering Done. Valid rows: {len(valid_df)}, Eliminated: {len(elim_df)}")

    # --- Step 2: Prepare Tensors ---
    X = valid_df.drop('label', axis=1).values
    y = valid_df['label'].values

    # --- Step 3: Launch Flower Client ---
    model = build_model(input_dim=X.shape[1])
    client = FlowerClient(model, X, y, trust_weights, client_id)
    
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=client
    )

if __name__ == "__main__":
    start_client()