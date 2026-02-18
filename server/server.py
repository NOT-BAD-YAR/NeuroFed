# server/server.py

import flwr as fl
import sys
import os

sys.dont_write_bytecode = True

# Ensure project root is on sys.path (same as original)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from strategy import SecureFedAvg
from model_loader import load_or_init_model
from flwr.common import ndarrays_to_parameters


def start_server():
    # 1) Load or initialize the global model
    model = load_or_init_model()
    initial_weights = model.get_weights()
    initial_parameters = ndarrays_to_parameters(initial_weights)

    # 2) Create strategy with initial parameters
    strategy = SecureFedAvg(
        min_fit_clients=1,       # change to 3 if you want 3 clients
        min_evaluate_clients=1,
        min_available_clients=1,
        initial_parameters=initial_parameters,
    )

    # 3) Start Flower server
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )


if __name__ == "__main__":
    start_server()
