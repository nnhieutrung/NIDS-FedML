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
    "ToN_IoT_50-50" : {
        "FEATURE_LABELS" : {
         "proto" : ['udp', 'icmp', 'tcp', ],
        "service" : ['-', 'http', 'ssl', 'ftp', 'gssapi', 'smb', 'dce_rpc', 'dns', 'smb;gssapi'],
        "conn_state" : ['OTH', 'SF', 'S0', 'S1', 'REJ', 'S3', 'SH', 'RSTRH', 'SHR', 'RSTO', 'RSTR', 'RSTOS0', 'S2'],
        "weird_name" : ['-','possible_split_routing', 'bad_TCP_checksum', 'bad_UDP_checksum', 'active_connection_reuse', 'data_before_established', 'inappropriate_FIN', 'above_hole_data_without_any_acks', 'DNS_RR_unknown_type', 'dnp3_corrupt_header_checksum', 'connection_originator_SYN_ack'],
        "weird_addl" : ['-','46','48'],
        "weird_notice": ['-','F'],
        "type" :  ['normal', 'backdoor', 'ddos', 'dos', 'injection', 'password', 'ransomware', 'scanning', 'xss', 'mitm'],
        },
        "INPUT_FEATURE" : ['proto', 'service', 'duration', 'src_bytes', 'dst_bytes',
           'conn_state', 'missed_bytes', 'src_pkts', 'src_ip_bytes', 'dst_pkts', 'dst_ip_bytes',
            'weird_name', 'weird_addl', 'weird_notice','label'],
        "OUTPUT_FEATURE" : 'type'
    },
    "ToN_IoT_80-20" : {
        "FEATURE_LABELS" : {
         "proto" : ['udp', 'icmp', 'tcp', ],
        "service" : ['-', 'http', 'ssl', 'ftp', 'gssapi', 'smb', 'dce_rpc', 'dns', 'smb;gssapi'],
        "conn_state" : ['OTH', 'SF', 'S0', 'S1', 'REJ', 'S3', 'SH', 'RSTRH', 'SHR', 'RSTO', 'RSTR', 'RSTOS0', 'S2'],
        "weird_name" : ['-','possible_split_routing', 'bad_TCP_checksum', 'bad_UDP_checksum', 'active_connection_reuse', 'data_before_established', 'inappropriate_FIN', 'above_hole_data_without_any_acks', 'DNS_RR_unknown_type', 'dnp3_corrupt_header_checksum', 'connection_originator_SYN_ack'],
        "weird_addl" : ['-','46','48'],
        "weird_notice": ['-','F'],
        "type" :  ['normal', 'backdoor', 'ddos', 'dos', 'injection', 'password', 'ransomware', 'scanning', 'xss', 'mitm'],
        },
        "INPUT_FEATURE" : ['proto', 'service', 'duration', 'src_bytes', 'dst_bytes',
           'conn_state', 'missed_bytes', 'src_pkts', 'src_ip_bytes', 'dst_pkts', 'dst_ip_bytes',
            'weird_name', 'weird_addl', 'weird_notice','label'],
        "OUTPUT_FEATURE" : 'type'
    },
    "ToN_IoT_95-5" : {
        "FEATURE_LABELS" : {
         "proto" : ['udp', 'icmp', 'tcp', ],
        "service" : ['-', 'http', 'ssl', 'ftp', 'gssapi', 'smb', 'dce_rpc', 'dns', 'smb;gssapi'],
        "conn_state" : ['OTH', 'SF', 'S0', 'S1', 'REJ', 'S3', 'SH', 'RSTRH', 'SHR', 'RSTO', 'RSTR', 'RSTOS0', 'S2'],
        "weird_name" : ['-','possible_split_routing', 'bad_TCP_checksum', 'bad_UDP_checksum', 'active_connection_reuse', 'data_before_established', 'inappropriate_FIN', 'above_hole_data_without_any_acks', 'DNS_RR_unknown_type', 'dnp3_corrupt_header_checksum', 'connection_originator_SYN_ack'],
        "weird_addl" : ['-','46','48'],
        "weird_notice": ['-','F'],
        "type" :  ['normal', 'backdoor', 'ddos', 'dos', 'injection', 'password', 'ransomware', 'scanning', 'xss', 'mitm'],
        },
        "INPUT_FEATURE" : ['proto', 'service', 'duration', 'src_bytes', 'dst_bytes',
           'conn_state', 'missed_bytes', 'src_pkts', 'src_ip_bytes', 'dst_pkts', 'dst_ip_bytes',
            'weird_name', 'weird_addl', 'weird_notice','label'],
        "OUTPUT_FEATURE" : 'type'
    },
    "ToN_IoT_unique" : {
        "FEATURE_LABELS" : {
         "proto" : ['udp', 'icmp', 'tcp', ],
        "service" : ['-', 'http', 'ssl', 'ftp', 'gssapi', 'smb', 'dce_rpc', 'dns', 'smb;gssapi'],
        "conn_state" : ['OTH', 'SF', 'S0', 'S1', 'REJ', 'S3', 'SH', 'RSTRH', 'SHR', 'RSTO', 'RSTR', 'RSTOS0', 'S2'],
        "weird_name" : ['-','possible_split_routing', 'bad_TCP_checksum', 'bad_UDP_checksum', 'active_connection_reuse', 'data_before_established', 'inappropriate_FIN', 'above_hole_data_without_any_acks', 'DNS_RR_unknown_type', 'dnp3_corrupt_header_checksum', 'connection_originator_SYN_ack'],
        "weird_addl" : ['-','46','48'],
        "weird_notice": ['-','F'],
        "type" :  ['normal', 'backdoor', 'ddos', 'dos', 'injection', 'password', 'ransomware', 'scanning', 'xss', 'mitm'],
        },
        "INPUT_FEATURE" : ['proto', 'service', 'duration', 'src_bytes', 'dst_bytes',
           'conn_state', 'missed_bytes', 'src_pkts', 'src_ip_bytes', 'dst_pkts', 'dst_ip_bytes',
            'weird_name', 'weird_addl', 'weird_notice','label'],
        "OUTPUT_FEATURE" : 'type'
    },
    "IoTiD20_50-50" : {
        "FEATURE_LABELS" : {
          'Cat' : ['Normal', 'Mirai', 'DoS', 'Scan', 'MITM ARP Spoofing'],
        },
        "INPUT_FEATURE" : ['Protocol', 'Flow_Duration', 'Tot_Fwd_Pkts', 'Tot_Bwd_Pkts',
       'TotLen_Fwd_Pkts', 'TotLen_Bwd_Pkts', 'Fwd_Pkt_Len_Max',
       'Fwd_Pkt_Len_Min', 'Fwd_Pkt_Len_Mean', 'Fwd_Pkt_Len_Std',
       'Bwd_Pkt_Len_Max', 'Bwd_Pkt_Len_Min', 'Bwd_Pkt_Len_Mean',
       'Bwd_Pkt_Len_Std', 'Flow_Byts/s', 'Flow_Pkts/s', 'Flow_IAT_Mean',
       'Flow_IAT_Std', 'Flow_IAT_Max', 'Flow_IAT_Min', 'Fwd_IAT_Tot',
       'Fwd_IAT_Mean', 'Fwd_IAT_Std', 'Fwd_IAT_Min', 'Bwd_IAT_Tot',
       'Bwd_IAT_Mean', 'Bwd_IAT_Std', 'Bwd_IAT_Min', 'Fwd_PSH_Flags',
       'Bwd_PSH_Flags', 'Fwd_URG_Flags', 'Bwd_URG_Flags', 'Fwd_Header_Len',
       'Bwd_Header_Len', 'Fwd_Pkts/s', 'Bwd_Pkts/s', 'Pkt_Len_Min',
       'Pkt_Len_Max', 'Pkt_Len_Mean', 'Pkt_Len_Std', 'Pkt_Len_Var',
       'FIN_Flag_Cnt', 'SYN_Flag_Cnt', 'RST_Flag_Cnt', 'ACK_Flag_Cnt',
       'URG_Flag_Cnt', 'CWE_Flag_Count', 'ECE_Flag_Cnt', 'Down/Up_Ratio',
        'Fwd_Byts/b_Avg', 'Fwd_Pkts/b_Avg', 'Fwd_Blk_Rate_Avg', 'Bwd_Byts/b_Avg',
       'Bwd_Pkts/b_Avg', 'Bwd_Blk_Rate_Avg', 'Init_Fwd_Win_Byts', 'Init_Bwd_Win_Byts',
       'Fwd_Act_Data_Pkts', 'Fwd_Seg_Size_Min', 'Active_Mean', 'Active_Std',
       'Active_Min', 'Idle_Mean', 'Idle_Std', 'Idle_Min'],
        "OUTPUT_FEATURE" : 'Cat'
    },
    
}