
from dataset.config import *

FEATURE_LABELS = None
INPUT_FEATURE = None
OUTPUT_FEATURE = None
DATASET = None
DATASET_PATH = None
MODEL_OUTSHAPE = None

LEARNING_RATE = 1e-3
BATCH_SIZE      = 256 #32
NUM_EPOCHS      = 10 #100

def get_dataset_config(dataset):
    global DATASET 
    global FEATURE_LABELS
    global INPUT_FEATURE
    global OUTPUT_FEATURE
    global DATASET_PATH 
    global MODEL_OUTSHAPE

    DATASET = dataset
    FEATURE_LABELS = DATASET_CONFIG[dataset]['FEATURE_LABELS']
    INPUT_FEATURE = DATASET_CONFIG[dataset]['INPUT_FEATURE']
    OUTPUT_FEATURE = DATASET_CONFIG[dataset]['OUTPUT_FEATURE']
    DATASET_PATH = './dataset/' + dataset + '/'
    MODEL_OUTSHAPE = len(FEATURE_LABELS[OUTPUT_FEATURE])

get_dataset_config('UNSW_NB15')

