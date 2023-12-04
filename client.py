import argparse
import os
from pathlib import Path
import tensorflow as tf
import flwr as fl
import pandas as pd
from sklearn.model_selection import train_test_split
import utils

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test, x_val, y_val):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.x_val, self.y_val = x_val, y_val
       
    def get_properties(self, config):
        """Get properties of client."""
        raise Exception("Not implemented")

    def get_parameters(self, config):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""
        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        round: int = config["round"]
        max_round: int = config["max_round"]
        
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        x_train, y_train = utils.split_dataset(self.x_train, self.y_train, round, max_round)
        x_val, y_val = utils.split_dataset(self.x_val, self.y_val, round, max_round)

        scaler = utils.get_scaler(config)
        x_train = scaler.transform(x_train)
        x_val = scaler.transform(x_val)

        class_weights = utils.calc_class_weights(y_train)
        y_train = utils.label_to_categorical(y_train)
        y_val = utils.label_to_categorical(y_val)

        # Train the model using hyperparameters from config
        history = self.model.fit(
            x_train,
            y_train,
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_val, y_val),
            class_weight=class_weights,
            workers=3
        )

 
        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(x_train)
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["acc"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_acc"][0],
        }
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        round: int = config["round"]
        max_round: int = config["max_round"]
        batch_size: int = config["batch_size"]

        x_test, y_test = utils.split_dataset(self.x_test, self.y_test, round, max_round)

        scaler = utils.get_scaler(config)
        x_test = scaler.transform(x_test)

        y_test = utils.label_to_categorical(y_test)

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(x_test, y_test, batch_size)
        num_examples_test = len(x_test)
        return loss, num_examples_test, {"accuracy": accuracy}


def main() -> None:
    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")

    parser.add_argument("--host", type=str,default="127.0.0.1")
    parser.add_argument("--dataset", type=int,default=0,choices=range(0,5))


    args = parser.parse_args()

    x_train, y_train = utils.get_dataset(df=utils.load_dataset_train() if args.dataset == 0 else utils.load_dataset_full(args.dataset))
    
    # Using A Part Train Data for Val
    x_val, y_val = utils.get_dataset(df=utils.load_dataset_train())
    _, x_val, _, y_val  = train_test_split(x_val, y_val, test_size=0.1)

    
    #Using A Part Test Data for Test
    x_test, y_test = utils.get_dataset(df=utils.load_dataset_test())
    _, x_test, _, y_test  = train_test_split(x_test, y_test, test_size=0.2)

    # Start Flower client
    model = utils.get_model(inshape=x_train.shape[1])

    client = CifarClient(model, x_train, y_train, x_test, y_test, x_val, y_val)

    fl.client.start_numpy_client(
        server_address= args.host + ":18922",
        client=client,
        root_certificates=Path(".cache/certificates/ca.crt").read_bytes(),
    )



if __name__ == "__main__":
    main()
