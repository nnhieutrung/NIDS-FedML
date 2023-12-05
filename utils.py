import models
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

MULTICLASS = True
LEARNING_RATE = 1e-3

FEATURE_LABELS = {
    "proto" :  ['3pc', 'a/n', 'aes-sp3-d', 'any', 'argus', 'aris', 'arp', 'ax.25', 'bbn-rcc', 'bna', 'br-sat-mon', 'cbt', 'cftp', 'chaos', 'compaq-peer', 'cphb', 'cpnx', 'crtp', 'crudp', 'dcn', 'ddp', 'ddx', 'dgp', 'egp', 'eigrp', 'emcon', 'encap', 'etherip', 'fc', 'fire', 'ggp', 'gmtp', 'gre', 'hmp', 'i-nlsp', 'iatp', 'ib', 'icmp', 'idpr', 'idpr-cmtp', 'idrp', 'ifmp', 'igmp', 'igp', 'il', 'ip', 'ipcomp', 'ipcv', 'ipip', 'iplt', 'ipnip', 'ippc', 'ipv6', 'ipv6-frag', 'ipv6-no', 'ipv6-opts', 'ipv6-route', 'ipx-n-ip', 'irtp', 'isis', 'iso-ip', 'iso-tp4', 'kryptolan', 'l2tp', 'larp', 'leaf-1', 'leaf-2', 'merit-inp', 'mfe-nsp', 'mhrp', 'micp', 'mobile', 'mtp', 'mux', 'narp', 'netblt', 'nsfnet-igp', 'nvp', 'ospf', 'pgm', 'pim', 'pipe', 'pnni', 'pri-enc', 'prm', 'ptp', 'pup', 'pvp', 'qnx', 'rdp', 'rsvp', 'rtp', 'rvd', 'sat-expak', 'sat-mon', 'sccopmce', 'scps', 'sctp', 'sdrp', 'secure-vmtp', 'sep', 'skip', 'sm', 'smp', 'snp', 'sprite-rpc', 'sps', 'srp', 'st2', 'stp', 'sun-nd', 'swipe', 'tcf', 'tcp', 'tlsp', 'tp++', 'trunk-1', 'trunk-2', 'ttp', 'udp', 'unas', 'uti', 'vines', 'visa', 'vmtp', 'vrrp', 'wb-expak', 'wb-mon', 'wsn', 'xnet', 'xns-idp', 'xtp', 'zero'],
    "service" : ['-', 'dhcp', 'dns', 'ftp', 'ftp-data', 'http', 'irc', 'pop3', 'radius', 'smtp', 'snmp', 'ssh', 'ssl'],
    "state" :  ['ACC', 'CLO', 'CON', 'ECO', 'FIN', 'INT', 'PAR', 'REQ', 'RST', 'URN', 'no'],
    "attack_cat" :  ['Analysis', 'Backdoor', 'DoS', 'Exploits', 'Fuzzers', 'Generic', 'Normal', 'Reconnaissance', 'Shellcode', 'Worms'],
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
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)



def get_outshape():
    return MODEL_OUTSHAPE 

def get_model(inshape: int, lr=LEARNING_RATE):
    model = models.model_conv1D(lr=lr,N=64,inshape=inshape,nclass=MODEL_OUTSHAPE)
    # model = models.model_baseline(lr=lr,inshape=inshape,nclass=MODEL_OUTSHAPE)
    # model = models.model_dense(lr=lr,N=64,inshape=inshape,nclass=MODEL_OUTSHAPE)
    # model = models.model_lstm(lr=lr,N=64,inshape=inshape,nclass=MODEL_OUTSHAPE)
    # model = models.model_conv1D_large(lr=lr,N=64,nfeat=inshape,nclass=MODEL_OUTSHAPE)
    # model = models.model_conv1D_binary(lr=lr,N=64,nfeat=inshape,nclass=MODEL_OUTSHAPE)
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
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}
    return class_weights

def label_to_categorical(y):
    return to_categorical(y, get_outshape())



def get_model_result(model, x, y, batch_size):
    pred = model.predict(x,batch_size=batch_size)

    f1 = f1_score(y.argmax(axis=-1), pred.argmax(axis=-1), average="weighted")
    precision = precision_score(y.argmax(axis=-1), pred.argmax(axis=-1), average="weighted")
    recall = recall_score(y.argmax(axis=-1), pred.argmax(axis=-1), average="weighted")
    accuracy = accuracy_score(y.argmax(axis=-1), pred.argmax(axis=-1))

    print("F1:" , f1_score(y.argmax(axis=-1), pred.argmax(axis=-1), average=None))
    print("Precision:" , precision_score(y.argmax(axis=-1), pred.argmax(axis=-1), average=None))
    print("Recall:" , recall_score(y.argmax(axis=-1), pred.argmax(axis=-1), average=None))
    print("Accuracy:" , accuracy)
    
    return {
        "f1" : f1,
        "precision" : precision,
        "recall" : recall,
        "accuracy" : accuracy,
    }

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


def load_datasets(idxs : list[int]):
    df = pd.concat([load_dataset_full(idx) for idx in idxs])    
    return df

def load_dataset_full(idx : int):
    print("loading dataset: ", idx)
    df = pd.read_csv("./dataset/UNSW_NB15-" + str(idx) + ".csv")
    return df



def split_dataset(x, y, round, max_round):
    if round == max_round:
        return x,y
    
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
        df[feature] = df[feature].map(lambda label : FEATURE_LABELS[feature].index(label))

    return df

