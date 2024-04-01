import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import *

# -------------------------
#       Dataset
# -------------------------
def get_dataset(df : pd.DataFrame):
    df = feature_labelencoding(df)

    x = df[['dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes',
       'dbytes', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt',
       'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt',
       'synack', 'ackdat', 'smean', 'dmean', 'trans_depth',
       'response_body_len', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm',
       'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm',
       'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm',
       'ct_srv_dst', 'is_sm_ips_ports']]
    y = df['attack_cat'] if MULTICLASS else df['label']
    return x.astype('float32'), y.astype('int32')

def load_dataset_train():
    df = pd.read_csv("./dataset/training-set.csv")
    return df

def load_dataset_validate():
    df = pd.read_csv("./dataset/validating-set.csv")
    return df

def load_dataset_test():
    df = pd.read_csv("./dataset/testing-set.csv")
    return df


def load_dataset_full(idx : int):
    print("loading dataset: ", idx)
    df = pd.read_csv("./dataset/training-set_" + str(idx) + ".csv")
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


def features_selection(df : pd.DataFrame):
    df = df[['dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports', 'attack_cat', 'label']]
    return df



def drop_sparse_columns(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    # BEGIN SOLUTION
    df = df.drop(columns=[col for col in df if df[col].isna().sum()/df.__len__() > threshold])
    return df
    # END SOLUTION
    pass


def get_feature_label(feature):
    return FEATURE_LABELS[feature]

def feature_labelencoding(df : pd.DataFrame):
    for feature in FEATURE_LABELS:
        df[feature] = [FEATURE_LABELS[feature].index(label) for label in df[feature]]

    return df


