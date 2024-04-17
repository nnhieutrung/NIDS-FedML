from utils import models
import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import compute_class_weight
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from keras.utils import to_categorical
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

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
    print("Get model for", dataset.get_dataset_path())
 
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
    pred = model.predict(x, batch_size=batch_size)

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



def plot_confussion_matrix(model, x_test, y_test, batch_size, path):
    categories = dataset.get_output_feature_labels()
    pred = model.predict(x_test, batch_size=batch_size)
    confusion  = tf.math.confusion_matrix(
        labels=y_test.argmax(axis=1),
        predictions=pred.argmax(axis=1),
        num_classes=len(categories)
    )
    
    conf_matrix = np.array(confusion)
    # print(conf_matrix)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha = 1)
    for i in range(conf_matrix.shape[0]):
        total = sum(conf_matrix[i])
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=f'{conf_matrix[i, j]}\n{conf_matrix[i, j]/total*100:.2f}%', va='center', ha='center', size='x-large')

    
    plt.xlabel('Predictions', fontsize=20)
    plt.ylabel('Actuals', fontsize=20)
    plt.xticks(np.arange(len(categories)), categories, rotation=26, fontsize=12)
    plt.yticks(np.arange(len(categories)), categories, fontsize=12)
    plt.title('Confusion Matrix', fontsize=30)
    plt.savefig(path, dpi = 300, bbox_inches = 'tight')
    

def plot_model_result(model, x, y, batch_size, path):
    pred = model.predict(x, batch_size=batch_size)
    # Data
    categories = dataset.get_output_feature_labels()
    f1_scores = f1_score(y.argmax(axis=-1), pred.argmax(axis=-1), average=None)
    precisions = precision_score(y.argmax(axis=-1), pred.argmax(axis=-1), average=None)
    recalls = recall_score(y.argmax(axis=-1), pred.argmax(axis=-1), average=None)

    # Transpose data
    data = np.array([f1_scores, precisions, recalls])

    # Plotting
    plt.figure(figsize=(10, 6))

    bar_width = 0.25
    index = np.arange(len(categories))
    
    plt.grid(axis='y', color = '#5A616E', linestyle = '--', linewidth = 0.5)

    plt.bar(index, data[0], bar_width, label='F1 Score', color='#3B74E5')
    plt.bar(index + bar_width, data[1], bar_width, label='Precision', color='#4CBD3B')
    plt.bar(index + 2*bar_width, data[2], bar_width, label='Recall', color='#F47710')

    plt.ylabel('Score', fontsize=14)
    plt.xlabel('Class', fontsize=14)
    plt.xticks(index + bar_width, categories, rotation=45, fontsize=12)
    plt.title('Performance Metrics for Each Class', fontsize=20)
    plt.legend()

    plt.tight_layout()
    plt.savefig(path, dpi = 300, bbox_inches = 'tight')