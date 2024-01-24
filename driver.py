import argparse
import socket
import threading
import torch
import flwr as fl
import logging
from collections import OrderedDict
from typing import Tuple, Dict, Optional
from strategy import CustomFedAvg
from utils.mappers import map_eval_metrics
from utils.training import evaluate
from utils.loaders import load_model, load_test_loader

logging.basicConfig(level=logging.INFO)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model(device)
test_loader = load_test_loader()
MIN_AVALIBLE_CLIENTS = 2

class Driver:
    def __init__(self, port, address=''):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((address, port))
        self.server.listen(50)
        self.server.settimeout(1)
        self.clients = []
        self.lock = threading.Lock()
        self.running = True

    def broadcast(self, message):
        for client in self.clients:
            client.send(message.encode())

    def handle_client(self, client_socket, client_address):
        while self.running:
            try:
                request = client_socket.recv(1024).decode()
                if request == "TRIGGER_FL":
                    with self.lock:
                        logging.info("Received TRIGGER_FL message.")
                        self.broadcast("FL_STARTED")
                        try:
                            start_flower_driver()
                        except Exception as e:
                            logging.error(f"FL round failed: {e}")
                            self.broadcast("FL_ERROR")
                        self.broadcast("FL_ENDED")
            except ConnectionResetError:
                logging.info('Client with the IP %s has DISCONECCTED', client_address[0])
                with self.lock:
                    self.clients.remove(client_socket)
                    if len(self.clients) < MIN_AVALIBLE_CLIENTS:
                        self.broadcast("WAITING")
                break

    def start(self):
        logging.info("Waiting for clients...")
        try:
            while self.running:
                try:
                    client_socket, client_address = self.server.accept()
                    logging.info('Client with the IP %s has CONNECTED', client_address[0])
                    self.clients.append(client_socket)

                    if len(self.clients) >= MIN_AVALIBLE_CLIENTS:  
                        self.broadcast("READY")
                    else:
                        client_socket.send("WAITING".encode())

                    client_thread = threading.Thread(target=self.handle_client, args=(client_socket, client_address))
                    client_thread.start()
                except socket.timeout:
                    continue
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        self.running = False
        for client in self.clients:
            client.close()
        self.server.close()

def start_flower_driver(server_address):
    logging.info("Triggering FL.")
    strategy = CustomFedAvg(
        fraction_fit=1.0,  
        min_available_clients=MIN_AVALIBLE_CLIENTS,
        evaluate_fn=centralized_evaluate,
    )

    history = fl.driver.start_driver(
        server_address=server_address,
        strategy=strategy,
    )

    logging.info("FL round finished.")
    print(history)

def centralized_evaluate(
    server_round: int,
    parameters: fl.common.NDArrays,
    config: Dict[str, fl.common.Scalar],
) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
    loss, _, metrics = evaluate(model, device, test_loader)
    return loss, map_eval_metrics(metrics)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--port', default=9093, type=int, help='Port')
    parser.add_argument('--server_ip', default='localhost', type=str, help='Server IP')
    parser.add_argument('--server_port', default=9091, type=int, help='Server port')

    args = parser.parse_args()

    server = Driver(args.port)
    try:
        server.start()
    except KeyboardInterrupt:
        server.stop()

    start_flower_driver(server_address=f"{args.server_ip}:{args.server_port}")