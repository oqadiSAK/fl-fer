import socket
import threading
from logging import INFO, ERROR
from fl.strategy import CustomFedAvg
from flwr.common.logger import log
from logging import INFO
import flwr as fl


class Driver:
    def __init__(self, port, server_address, min_available_clients, centralized_evaluate_fn, address=''):
        self.server_address = server_address
        self.min_available_clients = min_available_clients
        self.centralized_evaluate_fn = centralized_evaluate_fn
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
                        log(INFO, "Received TRIGGER_FL message.")
                        self.broadcast("FL_STARTED")
                        try:
                            accuracy = start_flower_driver(self.server_address, self.min_available_clients, self.centralized_evaluate_fn)
                            self.broadcast(f"ACCURACY {accuracy:.2f}")
                        except Exception as e:
                            log(ERROR, f"FL round failed: {e}")
                            self.broadcast("FL_ERROR")
                        self.broadcast("FL_ENDED")
            except ConnectionResetError:
                log(INFO, f'Client with the IP {client_address[0]} has DISCONNECTED')
                with self.lock:
                    self.clients.remove(client_socket)
                break

    def start(self):
        log(INFO, "Waiting for clients...")
        try:
            while self.running:
                try:
                    client_socket, client_address = self.server.accept()
                    log(INFO, f'Client with the IP {client_address[0]} has CONNECTED')
                    self.clients.append(client_socket)

                    if len(self.clients) >= self.min_available_clients:  
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
        
def start_flower_driver(server_address, min_available_clients, centralized_evaluate):
    log(INFO, "Triggering FL.")
    strategy = CustomFedAvg(
        fraction_fit=1.0,  
        min_available_clients=min_available_clients,
        evaluate_fn=centralized_evaluate,
    )

    history = fl.driver.start_driver(
        server_address=server_address,
        strategy=strategy,
    )

    log(INFO, "FL round finished.")
    print(history)
    last_accuracy = history.metrics_centralized['accuracy'][-1][1]
    return last_accuracy