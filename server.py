import argparse
from typing import Dict, Optional, Tuple
from pathlib import Path
import flwr as fl
from sklearn.preprocessing import MinMaxScaler
import utils



NUM_CLIENTS     = 3
BATCH_SIZE      = 16
NUM_EPOCHS      = 100
NUM_ROUNDS      = 10




def main() -> None:
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("-p", "--port", type=str,default="18922")
    parser.add_argument("-t", "--test", type=int,default=1,choices=range(1,9))
    

    args = parser.parse_args()

    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation
    x_test, y_test = utils.get_dataset(df=utils.load_dataset_full(args.test))


    scaler = MinMaxScaler()
    x_test = scaler.fit_transform(x_test)
    y_test = utils.label_to_categorical(y_test)
    
    utils.init_scaler(scaler)
    model = utils.get_model(inshape=x_test.shape[1])

      # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        accept_failures=False,
        evaluate_fn=get_evaluate_fn(model, x_test, y_test),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
    )

    # Start Flower server (SSL-enabled) for four rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:" + args.port,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        certificates=(
            Path(".cache/certificates/ca.crt").read_bytes(),
            Path(".cache/certificates/server.pem").read_bytes(),
            Path(".cache/certificates/server.key").read_bytes(),
        ),
    )


def get_evaluate_fn(model, x_test, y_test):
    """Return an evaluation function for server-side evaluation."""

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(parameters)  # Update model with the latest parameters
        result = utils.get_model_result(self.model, x_test, y_test, BATCH_SIZE)
        return loss, {"accuracy": accuracy}

    return evaluate

def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one local epoch,
    increase to two local epochs afterwards.
    """
    
    config = {
        "round" : server_round,
        "max_round" : NUM_ROUNDS,
        "batch_size": BATCH_SIZE,
        "local_epochs": NUM_EPOCHS,
    }

    config = utils.set_scaler(config)

    return config



def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five batches) during
    rounds one to three, then increase to ten local evaluation steps.
    """

     
    config = {
        "round" : server_round,
        "max_round" : NUM_ROUNDS,
        "batch_size": BATCH_SIZE,
        "local_epochs": NUM_EPOCHS,
    }
    
    config = utils.set_scaler(config)

    return config


if __name__ == "__main__":
    main()

