import hashlib
import numpy as np

def hash_model_weights(weights):
    """Creates a SHA-256 hash of the model weights (list of ndarrays)."""
    hash_obj = hashlib.sha256()
    for w in weights:
        hash_obj.update(w.tobytes())
    return hash_obj.hexdigest()