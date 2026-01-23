import flwr as fl
import sys
sys.dont_write_bytecode = True
import os

# This line ensures the root directory is in your path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from strategy import SecureFedAvg

def start_server():
    strategy = SecureFedAvg(
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3
    )
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy
    )

if __name__ == "__main__":
    start_server()
