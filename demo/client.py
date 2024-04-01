import argparse
import os
from pathlib import Path
import flwr as fl
from sklearn.model_selection import train_test_split
import utils

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

DATALIST = [1]

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
        split_dataset: bool = config["split_dataset"]

        x_train, y_train = self.x_train, self.y_train
        x_val, y_val = self.x_val, self.y_val

        if split_dataset:
            x_train, y_train = utils.split_dataset(x_train, y_train, round, max_round)
            x_val, y_val = utils.split_dataset(x_val, y_val, round, max_round)

        values = y_train.value_counts()

        labels = utils.get_feature_label('attack_cat')

        value_counts = {}
        for index, label in enumerate(labels):
            if index in values:
                value_counts[label] = values[index]
        
        print("Attack Types:", value_counts)
        
        with open('client_log', 'a') as file:
            # Write the line to the file
            file.write('%s - Round %s : %s \n' % (str(DATALIST), round, str(value_counts)))


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
            # validation_split = 0.2,
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
            "val_accuracy": history.history["val_acc"][0]
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
        split_dataset: bool = config["split_dataset"]
        
        x_test, y_test = self.x_test, self.y_test
        if split_dataset:
            x_test, y_test = utils.split_dataset(x_test, y_test, round, max_round)

        scaler = utils.get_scaler(config)
        x_test = scaler.transform(x_test)

        y_test = utils.label_to_categorical(y_test)

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy, f1, prec, recall = self.model.evaluate(x_test, y_test, batch_size)
        num_examples_test = len(x_test)
        return loss, num_examples_test, {"accuracy": accuracy}


def main() -> None:
    parser = argparse.ArgumentParser(description="Flower")

    parser.add_argument("--host", type=str,default="127.0.0.1")
    parser.add_argument("-p", "--port", type=str,default="18922")
    parser.add_argument("-d", "--data", type=int,nargs='+',default=1,choices=range(1,7))

    args = parser.parse_args()
    global DATALIST
    DATALIST = args.data if isinstance(args.data, list) else [args.data]
    x_train, y_train = utils.get_dataset(df=utils.load_datasets(DATALIST))
    
    x_val, y_val = utils.get_dataset(df=utils.load_dataset_validate())

    if len(x_val) > 0.1*len(x_train):
        _, x_val, _, y_val  = train_test_split(x_val, y_val, test_size=0.1*len(x_train)/len(x_val), stratify=y_val)


    x_test, y_test = utils.get_dataset(df=utils.load_dataset_test())
    _, x_test, _, y_test  = train_test_split(x_test, y_test, test_size=0.2, stratify=y_test)

    # Start Flower client
    model = utils.get_model(inshape=x_train.shape[1])

    client = CifarClient(model, x_train, y_train, x_test, y_test, x_val, y_val)

    fl.client.start_numpy_client(
        server_address= args.host + ":" + args.port,
        client=client,
        root_certificates=Path(".cache/certificates/ca.crt").read_bytes(),
    )



if __name__ == "__main__":
    main()
