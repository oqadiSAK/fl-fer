import flwr as fl
import torch
from collections import OrderedDict
from utils.training import train_without_validation
from utils.loaders import load_dynamic_train_loader

EPOCH_PER_ROUND = 2
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-3

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, device, participate_threshold):
        self.model = model
        self.device = device
        self.participate_threshold = participate_threshold
        
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        try:
            dynamic_training_loader = load_dynamic_train_loader(self.participate_threshold)
            len, _ = train_without_validation(self.model, self.device, dynamic_training_loader,
                                EPOCH_PER_ROUND, LEARNING_RATE, MOMENTUM, WEIGHT_DECAY)
            return self.get_parameters(config), len, {}
        except ValueError as e:
            print(e)
            return self.get_parameters(config), 0, {}
    
    # We dont want to evaluate the model on the client side as our edge devices are not powerful enough
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        return 0.0, 0, {}

def start_client(model, device, server_address, threshold):
    fl.client.start_numpy_client(
        server_address=server_address,  
        client=FlowerClient(model, device, threshold),
        transport="grpc-rere", 
    )