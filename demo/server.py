import argparse
from typing import Dict, Optional, Tuple
from pathlib import Path
import flwr as fl
from sklearn.preprocessing import MinMaxScaler
import utils
import os


NUM_CLIENTS     = 2
BATCH_SIZE      = 32
NUM_EPOCHS      = 100
NUM_ROUNDS      = 1
SPLIT_DATASET   = True



def main() -> None:
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("-p", "--port", type=str,default="18922")
    parser.add_argument("-nr", "--nround", type=int,default=4)
    parser.add_argument("-bs", "--bsize", type=int,default=64)
    parser.add_argument("-nl", "--nclient", type=int,default=2)
    parser.add_argument("-sd", "--sdataset", type=bool,default=True)

    args = parser.parse_args()
    
    global NUM_CLIENTS
    NUM_CLIENTS = args.nclient
    global SPLIT_DATASET
    SPLIT_DATASET = args.sdataset
    global NUM_ROUNDS
    NUM_ROUNDS = args.nround
    global BATCH_SIZE
    BATCH_SIZE = args.bsize

    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation
    x_test, y_test = utils.get_dataset(df=utils.load_dataset_test())


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
        loss, accuracy, f1, prec, recall  = model.evaluate(x_test, y_test, BATCH_SIZE)
        result = {"accuracy" : accuracy}
        result, details = utils.get_model_result(model, x_test, y_test, BATCH_SIZE)
        model.save('my_model.h5')

        with open('server_log', 'a') as file:
            file.write('Round %s - details : %s \n' % (server_round, str(details)))
            file.write('Round %s - result : %s \n' % (server_round, str(result)))
        return loss, result

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
        "split_dataset" : SPLIT_DATASET,
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
        "split_dataset" : SPLIT_DATASET,
    }
    
    config = utils.set_scaler(config)

    return config


if __name__ == "__main__":
    main()

