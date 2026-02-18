# server/model_loader.py

import os
import tensorflow as tf

from blockchain.blockchain import Blockchain
from utils.hash_utils import hash_model_weights
from strategy import build_simple_model  # local module in same folder

MODEL_PATH = "global_model.h5"


def load_or_init_model():
    """Load last global model if it matches blockchain, otherwise init fresh."""
    if os.path.exists(MODEL_PATH):
        print("🔁 Loading existing global_model.h5...")
        model = tf.keras.models.load_model(MODEL_PATH)

        # Compute hash of the loaded model
        weights = model.get_weights()
        current_hash = hash_model_weights(weights)

        # Load blockchain and get last block
        bc = Blockchain()
        latest = bc.chain[-1]

        print(f"[DEBUG] Loaded model hash:   {current_hash}")
        print(f"[DEBUG] Blockchain last hash:{latest.model_hash}")

        if current_hash != latest.model_hash:
            print("⚠️ WARNING: Saved model does NOT match blockchain, "
                  "starting from fresh model.")
            model = build_simple_model(input_dim=3)
        else:
            print("✅ Loaded model matches blockchain, continuing training.")
    else:
        print("✨ No saved global model found, initializing fresh model.")
        model = build_simple_model(input_dim=3)

    return model
