import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


from config import *
from dataset.config import *

FEATURE_LABELS = None
INPUT_FEATURE = None
OUTPUT_FEATURE = None
DATASET = None
DATASET_PATH = None
MODEL_OUTSHAPE = None



def get_feature_labels():
    return FEATURE_LABELS

def get_input_feature():
    return INPUT_FEATURE

def get_output_feature():
    return OUTPUT_FEATURE

def get_dataset():
    return DATASET

def get_dataset_path():
    return DATASET_PATH

def get_model_outshape():
    return MODEL_OUTSHAPE

def change_dataset(dataset):
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


change_dataset('UNSW_NB15')

# -------------------------
#       Dataset
# -------------------------

def get_dataset(df : pd.DataFrame):
    df = feature_labelencoding(df)

    x = df[INPUT_FEATURE]
    y = df[OUTPUT_FEATURE]
    return x.astype('float32'), y.astype('int32')

def load_dataset_train():
    df = pd.read_csv(DATASET_PATH + "training-set.csv")
    return df

def load_dataset_validate():
    df = pd.read_csv(DATASET_PATH + "validating-set.csv")
    return df

def load_dataset_test():
    df = pd.read_csv(DATASET_PATH + "testing-set.csv")
    return df

def load_dataset_full(idx : int):
    print("loading dataset: ", idx)
    df = pd.read_csv(DATASET_PATH + "training-set_" + str(idx) + ".csv")
    return df



def split_dataset(x, y, round, max_round):
    if round == max_round:
        return x,y
    
    # x_combined = split_list(x, round - 1, max_round)
    # y_combined = split_list(y, round - 1, max_round)
    _, x_combined, _, y_combined = train_test_split(x, y, test_size=round/max_round, random_state=42)
    return x_combined, y_combined
 

def split_list(lst, index, count):
    size = (int) (len(lst) // count)
    return lst[index * size : (index + 1) * size]



def normalize_dataframe(df : pd.DataFrame):
    """
    Trim whitespace from ends of each value across all series in dataframe
    """
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df = df.replace(r'^\s*$', np.nan, regex=True)
    return df



def drop_sparse_columns(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    # BEGIN SOLUTION
    df = df.drop(columns=[col for col in df if df[col].isna().sum()/df.__len__() > threshold])
    return df
    # END SOLUTION
    pass


def get_feature_label(feature):
    return FEATURE_LABELS[feature]


def get_output_feature_labels():
    return get_feature_label(OUTPUT_FEATURE)


def feature_labelencoding(df : pd.DataFrame):
    for feature in FEATURE_LABELS:
        df[feature] = [FEATURE_LABELS[feature].index(label) for label in df[feature]]

    return df


