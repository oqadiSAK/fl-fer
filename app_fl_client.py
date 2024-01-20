import flwr as fl
import torch
from collections import OrderedDict
from utils.mappers import map_eval_metrics, map_train_metrics
from utils.training import train ,evaluate
from utils.loaders import load_dynamic_train_loader, load_test_loader, load_validate_loader

EPOCH_PER_ROUND = 2
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-3

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, device, test_loader, val_loader):
        self.model = model
        self.device = device
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
        try:
            dynamic_training_loader = load_dynamic_train_loader()
            len, metrics = train(self.model, self.device, dynamic_training_loader, self.val_loader,
                                EPOCH_PER_ROUND, LEARNING_RATE, MOMENTUM, WEIGHT_DECAY)
            return self.get_parameters(config), len, map_train_metrics(metrics)
        except ValueError as e:
            print(e)
            return self.get_parameters(config), 0, {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, len, metrics = evaluate(self.model, self.device, self.test_loader)
        return loss, len, map_eval_metrics(metrics)

def start_client(model, device):
    test_loader = load_test_loader()
    validation_loader = load_validate_loader()
    
    # Start Flower client
    fl.client.start_numpy_client(
        server_address="192.168.1.102:9092",  
        client=FlowerClient(model, device, test_loader, validation_loader),
        transport="grpc-rere", 
    )