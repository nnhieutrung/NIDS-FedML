ETH_SERVER = 'HTTP://localhost:8545'
FLWR_PORT = '18922'
FLWR_SERVER = 'localhost'
FLWR_CLIEND_ID = None

LEARNING_RATE   = 1e-3
BATCH_SIZE      = 64 #32
NUM_EPOCHS      = 100 #100

CTGAN_NUM_EPOCHS = 300 #100
CTGAN_LENGTH    = 500
DATAFAKE_ETH_SIZE = 30000


import json
try: 
    print("Loading local config")
    
    with open('./local_config.json', 'r') as config_training:
        config=config_training.read()
        data = json.loads(config)
        for key, val in data.items():
            globals()[key] = val
     
except:
    print("Not found local config")