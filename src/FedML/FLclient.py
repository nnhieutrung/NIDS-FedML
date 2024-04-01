import argparse
import os
import hashlib
import numpy as np
import random
import tensorflow as tf
import flwr as fl

from services.bc_client_service import *
from utils.ipfs import *
from utils import utils
from utils import dataset


blockchainService = BlockchainService()


# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test, x_val, y_val, client_id, client_address):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.x_val, self.y_val = x_val, y_val
        self.client_id = client_id
        self.client_address = client_address

    def get_properties(self, config):
        """Get properties of client."""
        raise Exception("Not implemented")

    def get_parameters(self, config):
        # """Get parameters of the local model."""
        # raise Exception("Not implemented (server-side parameter initialization)")
        return self.model.get_weights()

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]
        session: int = config["session"]
        round: int = config["round"]
        max_round: int = config["max_round"]


        x_train, y_train = dataset.split_dataset(self.x_train, self.y_train, round, max_round)
        x_val, y_val = dataset.split_dataset(self.x_val, self.y_val, round, max_round)
        
        values = y_train.value_counts()

        labels = dataset.get_feature_label('attack_cat')

        value_counts = {}
        for index, label in enumerate(labels):
            if index in values:
                value_counts[label] = values[index]
        
        print("Attack Types:", value_counts)
        
        with open('./logs/client_log', 'a') as file:
            # Write the line to the file
            file.write('%s - Round %s : %s \n' % (self.client_id, round, str(value_counts)))


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
        num_examples_train = len(self.x_train)
        results = {
            "client_id": self.client_id,
            "loss": history.history["loss"][0],
            "accuracy": history.history["acc"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_acc"][0],
            "client_address": self.client_address,            
        }

        # Save training weights in the created directory
        if not (os.path.exists(f'./save-weights/Local-weights')):
            os.mkdir(f"./save-weights/Local-weights")

        if not (os.path.exists(f'./save-weights/Local-weights/Client-{self.client_id}')):
            os.mkdir(f"./save-weights/Local-weights/Client-{self.client_id}")

        if not (os.path.exists(f'./save-weights/Local-weights/Client-{self.client_id}/Session-{session}')):
            os.mkdir(f"./save-weights/Local-weights/Client-{self.client_id}/Session-{session}")       

        filename = f'./save-weights/Local-weights/Client-{self.client_id}/Session-{session}/Round-{round}-training-weights.npy'
        np.save(filename, parameters_prime)
        with open(filename,"rb") as f:
            bytes = f.read() # read entire file as bytes
            readable_hash = hashlib.sha256(bytes).hexdigest() #hash the file
            print(readable_hash)

        bcResult = blockchainService.addWeight(_session=session,_round_num=round, _dataSize=num_examples_train, _filePath = filename, _fileHash = readable_hash, client_id=self.client_id)
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        batch_size: int = config["batch_size"]
        session: int = config["session"]
        round: int = config["round"]
        max_round: int = config["max_round"]

        # Get global weights
        global_rnd_model = self.model.get_weights()


        x_test, y_test = dataset.split_dataset(self.x_test, self.y_test, round, max_round)

        scaler = utils.get_scaler(config)
        x_test = scaler.transform(x_test)

        y_test = utils.label_to_categorical(y_test)

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy, f1, prec, recall = self.model.evaluate(x_test, y_test, batch_size)
        num_examples_test = len(self.x_test)

        # Create directory for global weights
        try:
            if not (os.path.exists(f'./save-weights/Global-weights')):
                os.mkdir(f"./save-weights/Global-weights")

            if not (os.path.exists(f'./save-weights/Global-weights/Session-{session}')):
                os.mkdir(f"./save-weights/Global-weights/Session-{session}")

            filename = f'./save-weights/Global-weights/Session-{session}/Round-{round}-Global-weights.npy'
            if not (os.path.exists(filename)):
                np.save(filename, global_rnd_model)
        except NameError:
            print(NameError)

        return loss, num_examples_test, {"accuracy": accuracy}
