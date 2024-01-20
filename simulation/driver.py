import torch
import flwr as fl
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from collections import OrderedDict
from typing import List, Tuple, Dict, Optional
from flwr.common import Metrics
from utils.plot import plot_fl_centralized_metrics, plot_fl_distributed_evaluation_metrics, plot_fl_distributed_fit_metrics, plot_fl_losses
from utils.mappers import map_eval_metrics
from utils.training import evaluate
from utils.loaders import load_data_loaders, load_model

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    examples = [num_examples for num_examples, _ in metrics]

    # Multiply accuracy of each client by number of examples used
    train_losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
    train_accuracies = [
        num_examples * m["train_acc"] for num_examples, m in metrics
    ]
    val_losses = [num_examples * m["val_loss"] for num_examples, m in metrics]
    val_accuracies = [num_examples * m["val_acc"] for num_examples, m in metrics]

    # Aggregate and return custom metric (weighted average)
    return {
        "train_loss": sum(train_losses) / sum(examples),
        "train_acc": sum(train_accuracies) / sum(examples),
        "val_loss": sum(val_losses) / sum(examples),
        "val_acc": sum(val_accuracies) / sum(examples),
    }

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
_, test_loader, _ = load_data_loaders()

# Define strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,  
    min_available_clients=2,
    fit_metrics_aggregation_fn=weighted_average,
    evaluate_fn=centralized_evaluate,
    evaluate_metrics_aggregation_fn=evaluate_weighted_average,
)

# Start Flower server
history = fl.driver.start_driver(
    server_address="0.0.0.0:9091",
    config=fl.server.ServerConfig(num_rounds=50),
    strategy=strategy,
)

print(history)
plot_fl_losses(history)
plot_fl_distributed_fit_metrics(history)
plot_fl_distributed_evaluation_metrics(history)
plot_fl_centralized_metrics(history)