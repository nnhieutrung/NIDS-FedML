import argparse
import os
import hashlib
import numpy as np
import random
import tensorflow as tf
import flwr as fl
import pandas as pd
from io import StringIO
import torch
from ctgan import CTGAN
import sys

from config import *
from services.bc_client_service import *
from utils.ipfs import *
from utils import utils
from utils import dataset


blockchainService = BlockchainService()
ctgan = None

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.device(0))
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
        _round: int = config["round"]
        max_round: int = config["max_round"]
        enable_ctgan: bool = config["enable_ctgan"]

        x_train, y_train = dataset.split_dataset(self.x_train, self.y_train, _round, max_round)
        x_val, y_val = dataset.split_dataset(self.x_val, self.y_val, _round, max_round)
        
        values = y_train.value_counts()

        labels = dataset.get_output_feature_labels()

        value_counts = {}
        for index, label in enumerate(labels):
            if index in values:
                value_counts[label] = values[index]
        
        print("Attack Types:", value_counts)
        
        if not (os.path.exists(f'./results/Session-{session}')):
            os.mkdir(f"./results/Session-{session}") 
        
        with open(f'./results/Session-{session}/client_log', 'a') as file:
            # Write the line to the file
            file.write('%s - Round %s : %s \n' % (self.client_id, _round, str(value_counts)))


        # CTGAN
        if enable_ctgan:
            print("Using CTGAN")
            
            train_data = pd.concat([self.x_train, self.y_train], axis=1, join="inner")
            global ctgan
            
            for datatype, feature in enumerate(dataset.get_output_feature_labels()):
                maxRows = 0
                if datatype in values:
                    maxRows = int(values[datatype])
                print(f"{feature}: {maxRows}")

                CTGAN_maxRows, CTGAN_minRows = blockchainService.getCTGANMaxRows(_session=session, _roundNum=_round, _datatype=datatype, _numRows=maxRows, client_id=self.client_id)

                print(CTGAN_maxRows, CTGAN_minRows)
                if CTGAN_maxRows - CTGAN_minRows > 0:
                    if maxRows == CTGAN_maxRows:
                        print("CTGAN ready for make datafake")

                        if ctgan is None:
                            ctgan_data = train_data[1:1]
                            for val in train_data[dataset.get_output_feature()].unique():
                                val_data = train_data[train_data[dataset.get_output_feature()] == val].drop_duplicates()
                                if len(val_data) > CTGAN_LENGTH:
                                    val_data = val_data.sample(n=CTGAN_LENGTH, random_state=42)
                                ctgan_data = pd.concat([ctgan_data, val_data])

                            ctgan = CTGAN(verbose=True, epochs=CTGAN_NUM_EPOCHS)
                            ctgan.set_device('cuda')
                            ctgan.fit(ctgan_data, ctgan_data.columns)

                        datalength = CTGAN_maxRows-CTGAN_minRows
                        datafake = train_data[1:1] 

                        while datafake.shape[0] < datalength*2:
                            datafake = pd.concat([datafake, ctgan.sample(datalength)])
                            datafake = datafake[datafake[dataset.get_output_feature()] == datatype].drop_duplicates()

                        datafake = datafake.sample(n=datalength, random_state=42)

                        print(datafake[dataset.get_output_feature()].value_counts())

                        datafake=datafake.to_csv()
                        
                        print("sending datafake to blockchain")
                        for i in range(0, len(datafake), DATAFAKE_ETH_SIZE):
                            percent = i/len(datafake)
                            sys.stdout.write('\r')
                            sys.stdout.write("[%-20s] %d%% : %d/%d" % ('='*round(percent*21), round(percent*100), i, len(datafake)))
                            sys.stdout.flush()
                            blockchainService.sendCTGANDatafake(_session=session, _roundNum=_round, _datatype=datatype, _datafake=datafake[i:i+DATAFAKE_ETH_SIZE], _complete=(i+DATAFAKE_ETH_SIZE) >= len(datafake), client_id=self.client_id)
                        print("send complete")
                    
                    else:
                        print("CTGAN wait datafake from another node")
                        train_fake = blockchainService.getCTGANDatafake(_session=session, _roundNum=_round, _datatype=datatype)

                        train_fake = pd.read_csv(StringIO(train_fake))
                        
                        train_fake = train_fake[:(CTGAN_maxRows-maxRows)]

                        print("Received " + str(len(train_fake)) + " datafake")
                        x_train_fake, y_train_fake = dataset.get_xy_dataset(train_fake)
                        
                        x_train = pd.concat([x_train, x_train_fake])
                        y_train = pd.concat([y_train, y_train_fake])
            
            values = y_train.value_counts()
            labels = dataset.get_output_feature_labels()

            value_counts = {}
            for index, label in enumerate(labels):
                if index in values:
                    value_counts[label] = values[index]
            
            print("After CTGAN | Attack Types:", value_counts)

        print("Training with " + str(len(x_train)) + " rows")
        
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

        filename = f'./save-weights/Local-weights/Client-{self.client_id}/Session-{session}/Round-{_round}-training-weights.npy'

        np.save(filename, np.asanyarray(parameters_prime, dtype="object"))

        with open(filename,"rb") as f:
            bytes = f.read() # read entire file as bytes
            readable_hash = hashlib.sha256(bytes).hexdigest() #hash the file
            print(readable_hash)

        bcResult = blockchainService.addWeight(_session=session,_roundNum=_round, _dataSize=num_examples_train, _filePath = filename, _fileHash = readable_hash, client_id=self.client_id)
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


        return loss, num_examples_test, {"accuracy": accuracy}
