import torch
import flwr as fl
from collections import OrderedDict
from typing import List, Tuple, Dict, Optional
from flwr.common import Metrics
from utils.mappers import map_eval_metrics
from utils.training import evaluate
from utils.loaders import load_model, load_test_loader

def evaluate_weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    examples = [num_examples for num_examples, _ in metrics]

    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]

    # Aggregate and return custom metric (weighted average)
    return {
        "accuracy": sum(accuracies) / sum(examples),
    }

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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model(device)
test_loader = load_test_loader()

# Define strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,  
    min_available_clients=2,
    evaluate_fn=centralized_evaluate,
    evaluate_metrics_aggregation_fn=evaluate_weighted_average,
)

# Start Flower server
history = fl.driver.start_driver(
    server_address="192.168.1.102:9091",
    strategy=strategy,
)

print(history)