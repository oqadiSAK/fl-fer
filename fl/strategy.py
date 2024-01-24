import flwr as fl
import numpy as np
from functools import reduce
from logging import WARNING
from typing import Dict, List, Optional, Tuple, Union, override
from flwr.common import (
    FitRes,
    EvaluateRes,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy

class CustomFedAvg(fl.server.strategy.FedAvg):
    
    @override
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}
        
        if not self.accept_failures and failures:
            return None, {}

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(self._custom_aggregate(weights_results))

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1: 
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
    
        
    @override
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        return None, {}
    
    def _custom_aggregate(self, results: List[Tuple[NDArrays, int]]) -> NDArrays:
        num_examples_total = sum(num_examples for (_, num_examples) in results)
        if num_examples_total == 0:
            return results[0][0]

        weighted_weights = [
            [layer * num_examples for layer in weights] if num_examples != 0 else [0]*len(weights) for weights, num_examples in results
        ]

        weights_prime: NDArrays = [
            reduce(np.add, layer_updates) / num_examples_total
            for layer_updates in zip(*weighted_weights)
        ]
        return weights_prime
       
