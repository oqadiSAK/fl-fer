import argparse
import torch
import flwr as fl
from utils.loaders import load_model, load_test_loader
from fl.driver import Driver
from collections import OrderedDict
from typing import Tuple, Dict, Optional
from utils.mappers import map_eval_metrics
from utils.training import evaluate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model(device)
test_loader = load_test_loader()
MIN_AVAILABLE_CLIENTS = 2

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

    server = Driver(args.port, f"{args.server_ip}:{args.server_port}", MIN_AVAILABLE_CLIENTS, centralized_evaluate)
    try:
        server.start()
    except KeyboardInterrupt:
        server.stop()