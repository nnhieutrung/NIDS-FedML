import models
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from tensorflow.keras.utils import to_categorical

MULTICLASS = True
LEARNING_RATE = 1e-3

FEATURE_LABELS = {
    "proto" : ['udp', 'arp', 'tcp', 'igmp', 'ospf', 'sctp', 'gre', 'ggp', 'ip', 'ipnip', 'st2', 'argus', 'chaos', 'egp', 'emcon', 'nvp', 'pup', 'xnet', 'mux', 'dcn', 'hmp', 'prm', 'trunk-1', 'trunk-2', 'xns-idp', 'leaf-1', 'leaf-2', 'irtp', 'rdp', 'netblt', 'mfe-nsp', 'merit-inp', '3pc', 'idpr', 'ddp', 'idpr-cmtp', 'tp++', 'ipv6', 'sdrp', 'ipv6-frag', 'ipv6-route', 'idrp', 'mhrp', 'i-nlsp', 'rvd', 'mobile', 'narp', 'skip', 'tlsp', 'ipv6-no', 'any', 'ipv6-opts', 'cftp', 'sat-expak', 'ippc', 'kryptolan', 'sat-mon', 'cpnx', 'wsn', 'pvp', 'br-sat-mon', 'sun-nd', 'wb-mon', 'vmtp', 'ttp', 'vines', 'nsfnet-igp', 'dgp', 'eigrp', 'tcf', 'sprite-rpc', 'larp', 'mtp', 'ax.25', 'ipip', 'aes-sp3-d', 'micp', 'encap', 'pri-enc', 'gmtp', 'ifmp', 'pnni', 'qnx', 'scps', 'cbt', 'bbn-rcc', 'igp', 'bna', 'swipe', 'visa', 'ipcv', 'cphb', 'iso-tp4', 'wb-expak', 'sep', 'secure-vmtp', 'xtp', 'il', 'rsvp', 'unas', 'fc', 'iso-ip', 'etherip', 'pim', 'aris', 'a/n', 'ipcomp', 'snp', 'compaq-peer', 'ipx-n-ip', 'pgm', 'vrrp', 'l2tp', 'zero', 'ddx', 'iatp', 'stp', 'srp', 'uti', 'sm', 'smp', 'isis', 'ptp', 'fire', 'crtp', 'crudp', 'sccopmce', 'iplt', 'pipe', 'sps', 'ib', 'icmp', 'udt', 'rtp', 'esp'],    "service" : ['-', 'ftp', 'smtp', 'snmp', 'http', 'ftp-data', 'dns', 'ssh', 'radius', 'pop3', 'dhcp', 'ssl', 'irc'],
    "service" : ['-', 'http', 'ftp', 'ftp-data', 'smtp', 'pop3', 'dns', 'snmp', 'ssl', 'dhcp', 'irc', 'radius', 'ssh'],
    "state" : ['INT', 'FIN', 'REQ', 'ACC', 'CON', 'RST', 'CLO', 'URH', 'ECO', 'TXD', 'URN', 'no', 'PAR', 'MAS', 'TST', 'ECR'],
    "attack_cat" : ['Normal', 'Reconnaissance', 'Backdoor', 'DoS', 'Exploits', 'Analysis', 'Fuzzers', 'Worms', 'Shellcode', 'Generic', 'Backdoors'],
}
MODEL_OUTSHAPE = len(FEATURE_LABELS['attack_cat'])
# MODEL_OUTSHAPE = 20





def get_outshape():
    return MODEL_OUTSHAPE 

def get_model(inshape: int):
    model = models.model_conv1D(lr=LEARNING_RATE,N=64,inshape=inshape,nclass=MODEL_OUTSHAPE)
    # model = models.model_dense(lr=LEARNING_RATE,N=64,inshape=inshape,nclass=MODEL_OUTSHAPE)
    # model = models.model_lstm(lr=LEARNING_RATE,N=64,inshape=inshape,nclass=MODEL_OUTSHAPE)
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


def load_dataset_header():
    return pd.read_csv("./dataset/UNSW_NB15_features.csv")

def load_dataset_full(idx : int):
    assert idx in range(1,5)
    dfheader = load_dataset_header()
    header = [x.lower() for x in dfheader["Name"]] 
    df = pd.read_csv("./dataset/UNSW_NB15_" + str(idx) + ".csv")
    df.columns = header
    df = normalize_dataframe(df)
    df["attack_cat"].fillna("Normal", inplace = True)
    df["is_ftp_login"].fillna(0, inplace = True)
    df["ct_flw_http_mthd"].fillna(0, inplace = True)
    return df

def load_dataset_train():
    df = pd.read_csv("./dataset/UNSW_NB15_training.csv")
    return df

def load_dataset_test():
    df = pd.read_csv("./dataset/UNSW_NB15_testing.csv")
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
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    df = df.replace(r'^\s*$', np.nan, regex=True)
    return df


def features_selection(df : pd.DataFrame):
    df = df[['dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports', 'attack_cat', 'label']]
    return df





def get_feature_label(feature):
    return FEATURE_LABELS[feature]

def feature_labelencoding(df : pd.DataFrame):
    for feature in FEATURE_LABELS:
        df[feature] = df[feature].map(lambda label : FEATURE_LABELS[feature].index(label))

    return df

