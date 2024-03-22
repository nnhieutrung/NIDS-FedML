import models
import json
import numpy as np
import pandas as pd
from ctgan import CTGAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from keras.utils import to_categorical


import tensorflow as tf

MULTICLASS = True
LEARNING_RATE = 1e-3

FEATURE_LABELS = {
    "proto" : ['udp', 'arp', 'tcp', 'igmp', 'ospf', 'sctp', 'gre', 'ggp', 'ip', 'ipnip', 'st2', 'argus', 'chaos', 'egp', 'emcon', 'nvp', 'pup', 'xnet', 'mux', 'dcn', 'hmp', 'prm', 'trunk-1', 'trunk-2', 'xns-idp', 'leaf-1', 'leaf-2', 'irtp', 'rdp', 'netblt', 'mfe-nsp', 'merit-inp', '3pc', 'idpr', 'ddp', 'idpr-cmtp', 'tp++', 'ipv6', 'sdrp', 'ipv6-frag', 'ipv6-route', 'idrp', 'mhrp', 'i-nlsp', 'rvd', 'mobile', 'narp', 'skip', 'tlsp', 'ipv6-no', 'any', 'ipv6-opts', 'cftp', 'sat-expak', 'ippc', 'kryptolan', 'sat-mon', 'cpnx', 'wsn', 'pvp', 'br-sat-mon', 'sun-nd', 'wb-mon', 'vmtp', 'ttp', 'vines', 'nsfnet-igp', 'dgp', 'eigrp', 'tcf', 'sprite-rpc', 'larp', 'mtp', 'ax.25', 'ipip', 'aes-sp3-d', 'micp', 'encap', 'pri-enc', 'gmtp', 'ifmp', 'pnni', 'qnx', 'scps', 'cbt', 'bbn-rcc', 'igp', 'bna', 'swipe', 'visa', 'ipcv', 'cphb', 'iso-tp4', 'wb-expak', 'sep', 'secure-vmtp', 'xtp', 'il', 'rsvp', 'unas', 'fc', 'iso-ip', 'etherip', 'pim', 'aris', 'a/n', 'ipcomp', 'snp', 'compaq-peer', 'ipx-n-ip', 'pgm', 'vrrp', 'l2tp', 'zero', 'ddx', 'iatp', 'stp', 'srp', 'uti', 'sm', 'smp', 'isis', 'ptp', 'fire', 'crtp', 'crudp', 'sccopmce', 'iplt', 'pipe', 'sps', 'ib', 'icmp', 'udt', 'rtp', 'esp'],    "service" : ['-', 'ftp', 'smtp', 'snmp', 'http', 'ftp-data', 'dns', 'ssh', 'radius', 'pop3', 'dhcp', 'ssl', 'irc'],
    "service" : ['-', 'http', 'ftp', 'ftp-data', 'smtp', 'pop3', 'dns', 'snmp', 'ssl', 'dhcp', 'irc', 'radius', 'ssh'],
    "state" : ['INT', 'FIN', 'REQ', 'ACC', 'CON', 'RST', 'CLO', 'URH', 'ECO', 'TXD', 'URN', 'no', 'PAR', 'MAS', 'TST', 'ECR'],
    "attack_cat" :  ['Normal', 'Generic', 'Exploits', 'Reconnaissance', 'Fuzzers', 'DoS', 'Shellcode', 'Analysis', 'Backdoor', 'Worms'],
}
MODEL_OUTSHAPE = len(FEATURE_LABELS['attack_cat'])
# MODEL_OUTSHAPE = 20

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


def get_outshape():
    return MODEL_OUTSHAPE 

def get_model(inshape: int, lr=LEARNING_RATE):
    # model = models.model_test(lr=lr,inshape=inshape,nclass=MODEL_OUTSHAPE)
    model = models.model_conv1D(lr=lr,N=64,inshape=inshape,nclass=MODEL_OUTSHAPE)
    # model = models.model_test(lr=lr,inshape=inshape,nclass=MODEL_OUTSHAPE)
    # model = models.model_dense(lr=lr,N=64,inshape=inshape,nclass=MODEL_OUTSHAPE)
    # model = models.model_lstm(lr=lr,N=64,inshape=inshape,nclass=MODEL_OUTSHAPE)
    # model = models.model_conv1D_large(lr=lr,nfeat=inshape,nclass=MODEL_OUTSHAPE)
    # model = models.model_conv1D_binary(lr=lr,nfeat=inshape,nclass=MODEL_OUTSHAPE)
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

    labels = get_feature_label('attack_cat')
    for i in range(len(labels)):
        if i not in class_weights:
            class_weights[i] = 0

    return class_weights

def label_to_categorical(y):
    return to_categorical(y, get_outshape())



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
    for i, feature in enumerate(FEATURE_LABELS['attack_cat']):
        details = details + '%s : f1 = %s, precision = %s, recall = %s \n' % (feature, features_f1[i], features_prec[i], features_recall[i])
        
    print(details)
    
    print("Accuracy:" , accuracy)
    
    return {
        "f1" : f1,
        "precision" : precision,
        "recall" : recall,
        "accuracy" : accuracy,
    }, details

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


def load_datasets(idxs : list[str]):
    df = pd.concat([load_dataset_full(idx) for idx in idxs])    
    return df

def load_dataset_train():
    df = pd.read_csv("./dataset/UNSW_NB15_training-set.csv")
    return df

def load_dataset_validate():
    df = pd.read_csv("./dataset/UNSW_NB15_validating-set.csv")
    return df

def load_dataset_test():
    df = pd.read_csv("./dataset/UNSW_NB15_testing-set.csv")
    return df


def load_dataset_full(idx : int):
    print("loading dataset: ", idx)
    df = pd.read_csv("./dataset/UNSW_NB15-" + str(idx) + ".csv")
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


