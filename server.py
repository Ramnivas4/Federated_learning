import flwr as fl
from typing import Dict, List, Tuple

# Define metric aggregation functions
def weighted_average(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    total_samples = sum(num_samples for num_samples, _ in metrics)
    weighted_metrics = {
        key: sum(num_samples * m[key] for num_samples, m in metrics) / total_samples
        for key in metrics[0][1]
    }
    return weighted_metrics

# Define a federated averaging strategy with metric aggregation
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,  
    fraction_evaluate=1.0,  
    min_fit_clients=1,  
    min_evaluate_clients=1,  
    min_available_clients=1,  
    evaluate_metrics_aggregation_fn=weighted_average,  # Aggregate evaluation metrics
    fit_metrics_aggregation_fn=weighted_average,  # Aggregate training metrics
)

if __name__ == "__main__":
    hist = fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=5),  # Increase training rounds
        strategy=strategy,
    )
    