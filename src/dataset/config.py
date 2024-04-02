DATASET_CONFIG = {
    "UNSW_NB15" : {
        "FEATURE_LABELS" : {
            "proto" : ['udp', 'arp', 'tcp', 'igmp', 'ospf', 'sctp', 'gre', 'ggp', 'ip', 'ipnip', 'st2', 'argus', 'chaos', 'egp', 'emcon', 'nvp', 'pup', 'xnet', 'mux', 'dcn', 'hmp', 'prm', 'trunk-1', 'trunk-2', 'xns-idp', 'leaf-1', 'leaf-2', 'irtp', 'rdp', 'netblt', 'mfe-nsp', 'merit-inp', '3pc', 'idpr', 'ddp', 'idpr-cmtp', 'tp++', 'ipv6', 'sdrp', 'ipv6-frag', 'ipv6-route', 'idrp', 'mhrp', 'i-nlsp', 'rvd', 'mobile', 'narp', 'skip', 'tlsp', 'ipv6-no', 'any', 'ipv6-opts', 'cftp', 'sat-expak', 'ippc', 'kryptolan', 'sat-mon', 'cpnx', 'wsn', 'pvp', 'br-sat-mon', 'sun-nd', 'wb-mon', 'vmtp', 'ttp', 'vines', 'nsfnet-igp', 'dgp', 'eigrp', 'tcf', 'sprite-rpc', 'larp', 'mtp', 'ax.25', 'ipip', 'aes-sp3-d', 'micp', 'encap', 'pri-enc', 'gmtp', 'ifmp', 'pnni', 'qnx', 'scps', 'cbt', 'bbn-rcc', 'igp', 'bna', 'swipe', 'visa', 'ipcv', 'cphb', 'iso-tp4', 'wb-expak', 'sep', 'secure-vmtp', 'xtp', 'il', 'rsvp', 'unas', 'fc', 'iso-ip', 'etherip', 'pim', 'aris', 'a/n', 'ipcomp', 'snp', 'compaq-peer', 'ipx-n-ip', 'pgm', 'vrrp', 'l2tp', 'zero', 'ddx', 'iatp', 'stp', 'srp', 'uti', 'sm', 'smp', 'isis', 'ptp', 'fire', 'crtp', 'crudp', 'sccopmce', 'iplt', 'pipe', 'sps', 'ib', 'icmp', 'udt', 'rtp', 'esp'],    "service" : ['-', 'ftp', 'smtp', 'snmp', 'http', 'ftp-data', 'dns', 'ssh', 'radius', 'pop3', 'dhcp', 'ssl', 'irc'],
            "service" : ['-', 'http', 'ftp', 'ftp-data', 'smtp', 'pop3', 'dns', 'snmp', 'ssl', 'dhcp', 'irc', 'radius', 'ssh'],
            "state" : ['INT', 'FIN', 'REQ', 'ACC', 'CON', 'RST', 'CLO', 'URH', 'ECO', 'TXD', 'URN', 'no', 'PAR', 'MAS', 'TST', 'ECR'],
            "attack_cat" :  ['Normal', 'Generic', 'Exploits', 'Reconnaissance', 'Fuzzers', 'DoS', 'Shellcode', 'Analysis', 'Backdoor', 'Worms'],
        },
        "INPUT_FEATURE" : ['dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes',
            'dbytes', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt',
            'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt',
            'synack', 'ackdat', 'smean', 'dmean', 'trans_depth',
            'response_body_len', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm',
            'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm',
            'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm',
            'ct_srv_dst', 'is_sm_ips_ports'
        ],
        "OUTPUT_FEATURE" : 'attack_cat'
    },
}