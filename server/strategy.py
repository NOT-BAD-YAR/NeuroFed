import flwr as fl
import tensorflow as tf
import os
import sys
from flwr.common import parameters_to_ndarrays
from utils.hash_utils import hash_model_weights
from blockchain.blockchain import Blockchain

# Initialize the blockchain at the module level
server_blockchain = Blockchain()

def build_simple_model(input_dim):
    """
    Local helper to reconstruct the model architecture 
    without needing to import from the client folder.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_dim=input_dim),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

class SecureFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        # 1. Call the base FedAvg to get the averaged weights
        aggregated = super().aggregate_fit(rnd, results, failures)
        
        if aggregated is not None:
            parameters, _ = aggregated
            
            # 2. Convert Parameters to NumPy arrays for Hashing and Saving
            weights = parameters_to_ndarrays(parameters)
            
            # 3. Create the Model Hash for the Blockchain
            model_hash = hash_model_weights(weights)
            
            # 4. Commit to Blockchain
            metadata = {
                "client_count": len(results),
                "failure_count": len(failures)
            }
            server_blockchain.add_block(rnd, model_hash, metadata)
            
            # --- MODEL EVOLUTION SAVING ---
            print(f"💾 Saving Round {rnd} Global Model to global_model.h5...")
            try:
                # We use 3 features: age, heart_rate, blood_pressure
                temp_model = build_simple_model(input_dim=3) 
                temp_model.set_weights(weights)
                
                # Save as .h5 for your model_test.py script
                temp_model.save("global_model.h5")
                print(f"✅ Round {rnd} committed to blockchain and saved to disk.")
            except Exception as e:
                print(f"⚠️ Error saving model file: {e}")
            
            # Print the chain for the audit demo
            server_blockchain.print_chain()
            
        return aggregated