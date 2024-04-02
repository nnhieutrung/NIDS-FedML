from utils import models
import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import compute_class_weight
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from keras.utils import to_categorical
import tensorflow as tf

from config import *
from utils import dataset

print("Tenserflow Version: ", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)


def get_model(inshape: int, lr=LEARNING_RATE):
    # model = models.model_test(lr=lr,inshape=inshape,nclass=dataset.get_model_outshape())
    model = models.model_conv1D(lr=lr,N=64,inshape=inshape,nclass=dataset.get_model_outshape())
    # model = models.model_test(lr=lr,inshape=inshape,nclass=dataset.get_model_outshape())
    # model = models.model_dense(lr=lr,N=64,inshape=inshape,nclass=dataset.get_model_outshape())
    # model = models.model_lstm(lr=lr,N=64,inshape=inshape,nclass=dataset.get_model_outshape())
    # model = models.model_conv1D_large(lr=lr,nfeat=inshape,nclass=dataset.get_model_outshape())
    # model = models.model_conv1D_binary(lr=lr,nfeat=inshape,nclass=dataset.get_model_outshape())
    return model



# -------------------------
#       Utils
# -------------------------

scaler_data = []


def init_scaler(scaler : MinMaxScaler):
    global scaler_data
    scaler_data = [scaler.data_min_.tolist(), scaler.data_max_.tolist()]

def set_scaler(config):
    config['scaler_data'] = json.dumps(scaler_data)

    return config

def get_scaler(config):
    
    scaler_data = json.loads(config['scaler_data'])

    scaler_data[0] = np.array(scaler_data[0]).reshape(-1, )
    scaler_data[1] = np.array(scaler_data[1]).reshape(-1, )

    scaler = MinMaxScaler()
    scaler.fit(scaler_data)
    return scaler

def calc_class_weights(y):
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
    class_weights = dict(zip(np.unique(y), class_weights))

    labels = dataset.get_output_feature_labels()
    for i in range(len(labels)):
        if i not in class_weights:
            class_weights[i] = 0

    return class_weights

def label_to_categorical(y):
    return to_categorical(y, dataset.get_model_outshape())



def get_model_result(model, x, y, batch_size):
    pred = model.predict(x,batch_size=batch_size)

    f1 = f1_score(y.argmax(axis=-1), pred.argmax(axis=-1), average="weighted")
    precision = precision_score(y.argmax(axis=-1), pred.argmax(axis=-1), average="weighted")
    recall = recall_score(y.argmax(axis=-1), pred.argmax(axis=-1), average="weighted")
    accuracy = accuracy_score(y.argmax(axis=-1), pred.argmax(axis=-1))

    features_f1 = f1_score(y.argmax(axis=-1), pred.argmax(axis=-1), average=None)
    features_prec = precision_score(y.argmax(axis=-1), pred.argmax(axis=-1), average=None)
    features_recall = recall_score(y.argmax(axis=-1), pred.argmax(axis=-1), average=None)

    details = ''
    for i, feature in enumerate(dataset.get_output_feature_labels()):
        details = details + '%s : f1 = %s, precision = %s, recall = %s \n' % (feature, features_f1[i], features_prec[i], features_recall[i])
        
    print(details)
    
    print("Accuracy:" , accuracy)
    
    return {
        "f1" : f1,
        "precision" : precision,
        "recall" : recall,
        "accuracy" : accuracy,
    }, details

