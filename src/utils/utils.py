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
import psutil
import time
import matplotlib.pyplot as plt
from threading import Thread, Event

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
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    norm_conf_matrix = conf_matrix / row_sums * 100
    # print(conf_matrix)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.matshow(norm_conf_matrix, cmap=plt.cm.Blues, alpha = 1)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=f'{conf_matrix[i, j]}\n{norm_conf_matrix[i, j]:.2f}%', va='center', ha='center', size='x-large')

    
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
    
    plt.bar(index, data[0], bar_width, label='F1 Score', color='#3B74E5')
    plt.bar(index + bar_width, data[1], bar_width, label='Precision', color='#4CBD3B')
    plt.bar(index + 2*bar_width, data[2], bar_width, label='Recall', color='#F47710')

    plt.ylabel('Score', fontsize=14)
    plt.xlabel('Class', fontsize=14)
    plt.xticks(index + bar_width, categories, rotation=45, fontsize=12)
    plt.ylim(0, 1.01)
    plt.yticks([0,0.2,0.4,0.6,0.8,1.0])
    plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], minor=True)
    plt.grid(which='both')
    plt.grid(axis='y', color = '#5A616E', linewidth = 0.5, alpha=0.8, which='major')
    plt.grid(axis='y', color = '#5A616E', linestyle = '--', linewidth = 0.5, alpha=0.5, which='minor')
    plt.grid(axis='x', alpha=0.0)


    plt.title('Performance Metrics for Each Class', fontsize=20)
    plt.legend()

    plt.tight_layout()
    plt.savefig(path, dpi = 300, bbox_inches = 'tight')


def monitor_resources(interval_sec, cpu_usage, ram_usage, stop_event):
    while not stop_event.is_set():
        cpu_percent = psutil.cpu_percent(interval=interval_sec)
        ram_amount = round(psutil.virtual_memory().used / (1024 ** 2))
        cpu_usage.append(cpu_percent)
        ram_usage.append(ram_amount)
        
def record_performance(interval_sec=1):
    cpu_usage = []
    ram_usage = []
    stop_event = Event()
    worker = Thread(target=monitor_resources, args=(interval_sec, cpu_usage, ram_usage, stop_event), daemon=True)
    worker.start()

    def get_result():
        stop_event.set()
        worker.join()
        return cpu_usage, ram_usage, interval_sec

    return get_result

def plot_performance_report(record, path):
    cpu_usage, ram_usage, record_interval = record()

    length = len(cpu_usage)

    min = len(cpu_usage) * record_interval // 60
    # Rescale data to fit within 1000-2000 points
    interval = max(1, length // 1000)  # Adjusted interval for CPU usage
    cpu_usage = cpu_usage[::interval]
    ram_usage = ram_usage[::interval]

    time_intervals = np.arange(0,length,interval)*record_interval/60

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(time_intervals, cpu_usage)
    plt.xticks(np.arange(0, min, 1.0))
    plt.ylim(0, 100)
    plt.title('CPU Usage Over Time')
    plt.grid(True)
    plt.ylabel('Usage (%)')
    plt.xticks(rotation=45)

    plt.subplot(2, 1, 2)
    plt.plot(time_intervals, ram_usage, color='orange')
    plt.xticks(np.arange(0, min, 1.0))
    plt.title('RAM Usage Over Time')
    plt.grid(True)
    plt.ylabel('Usage (MB)')
    plt.xticks(rotation=45)

    plt.xlabel('Time')
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.show()
