import flwr as fl
import logging
logging.basicConfig(level=logging.DEBUG)


# Define the Flower server
server = fl.server.Server(client_manager=fl.server.SimpleClientManager(), strategy=fl.server.strategy.FedAvg())

# Start the Flower server
logging.debug("Starting server")
fl.server.start_server(server_address="127.0.0.1:8000", server=server, config=fl.server.ServerConfig(num_rounds=2,round_timeout=600))
logging.debug("Flower server closed.")