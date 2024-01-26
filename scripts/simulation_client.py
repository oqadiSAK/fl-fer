import flwr as fl
import torch
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from collections import OrderedDict
from utils.mappers import map_eval_metrics, map_train_metrics
from utils.training import train ,evaluate
from utils.loaders import load_data_loaders, load_model

EPOCH_PER_ROUND = 2
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-3

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, device, training_loader, test_loader, val_loader):
        self.model = model
        self.device = device
        self.training_loader = training_loader
        self.test_loader = test_loader
        self.val_loader = val_loader

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        len, metrics = train(self.model, self.device, self.training_loader, self.val_loader,
                            EPOCH_PER_ROUND, LEARNING_RATE, MOMENTUM, WEIGHT_DECAY)
        
        return self.get_parameters(config), len, map_train_metrics(metrics)

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, len, metrics = evaluate(self.model, self.device, self.test_loader)
        return loss, len, map_eval_metrics(metrics)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(device)
    training_loader, test_loader, validation_loader = load_data_loaders()
    
    # Start Flower client
    fl.client.start_numpy_client(
        server_address="localhost:9092",  
        client=FlowerClient(model, device, training_loader, test_loader, validation_loader),
        transport="grpc-rere", 
    )

if __name__ == "__main__":
    main()