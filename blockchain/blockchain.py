import time
import hashlib
import json
import os

class Block:
    # Added **kwargs to capture 'hash' and any other extra fields from JSON
    def __init__(self, index, model_hash, prev_hash, metadata=None, timestamp=None, **kwargs):
        self.index = index
        self.timestamp = timestamp or time.time()
        self.model_hash = model_hash
        self.prev_hash = prev_hash
        self.metadata = metadata or {}
        
        # We re-calculate the hash rather than trusting the one in the JSON.
        # This is a key security feature: "Verification on Load"
        self.hash = self.compute_hash()

    def compute_hash(self):
        # Ensure the hash is generated from the core data
        data = f"{self.index}{self.timestamp}{self.model_hash}{self.prev_hash}{self.metadata}"
        return hashlib.sha256(data.encode()).hexdigest()

class Blockchain:
    def __init__(self, file_path="blockchain_ledger.json"):
        self.file_path = file_path
        self.chain = []
        if os.path.exists(self.file_path):
            self._load_from_disk()
        else:
            self._create_genesis()

    def _create_genesis(self):
        genesis = Block(0, "GENESIS_ROOT", "0", {"info": "Network Start"})
        self.chain.append(genesis)
        self.save()

    def add_block(self, round_no, model_hash, metadata=None):
        if metadata is None:
            metadata = {}
        # Ensure the chain is not empty before accessing index -1
        prev_hash = self.chain[-1].hash if self.chain else "0"
        new_block = Block(round_no, model_hash, prev_hash, metadata)
        self.chain.append(new_block)
        self.save()

    def save(self):
        # We use __dict__ which saves all properties including the hash
        with open(self.file_path, "w") as f:
            json.dump([b.__dict__ for b in self.chain], f, indent=4)

    def _load_from_disk(self):
        try:
            with open(self.file_path, "r") as f:
                data = json.load(f)
                # The **b unpacking now works because Block.__init__ accepts **kwargs
                self.chain = [Block(**b) for b in data]
        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠️ Warning: Ledger file corrupted or incompatible. Resetting chain. Error: {e}")
            self.chain = []
            self._create_genesis()

    def print_chain(self):
        print("\n--- Blockchain Ledger ---")
        for b in self.chain:
            print(f"Round {b.index} | Hash: {b.hash[:10]}... | Metadata: {b.metadata}")
        print("--------------------------\n")