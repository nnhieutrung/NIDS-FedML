import os
import json
import time
import numpy as np

import flwr as fl
from fastapi import FastAPI, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from sklearn.preprocessing import MinMaxScaler

from FedML.FLstrategy import *
from services.bc_server_service import BlockchainService

from utils import utils
from utils import dataset
from dataset.config import *

app=FastAPI()
blockchainService = BlockchainService()


@app.get('/getContributions')
def getContributions():
    contributions = blockchainService.getContributions()
    # Conver Python list to JSON
    json_compatible_item_data = jsonable_encoder(contributions)
    return JSONResponse(content=json_compatible_item_data)


@app.get('/getTrainingSessions')
def getTrainingSessions():
    trainingSessions = blockchainService.getTrainingSessions()
    # Conver Python list to JSON
    json_compatible_item_data = jsonable_encoder(trainingSessions)
    return JSONResponse(content=json_compatible_item_data)


    
@app.post("/launchFL")
def launch_fl_session(num_rounds:int= Query(4), num_clients:int= Query(2), is_resume:bool = Query(False), budget: float = Query(100), dataset_name: str = Query(enum=[key for key in DATASET_CONFIG]), enable_ctgan: bool = Query(True)):
    """Start server and trigger update_strategy then connect to clients to perform fl session"""
    print(dataset_name, is_resume)
    if dataset_name not in DATASET_CONFIG:
        return {"error": "Invalid dataset name"}
    
    dataset.change_dataset(dataset_name)
    print(dataset.get_dataset_path())
 
 
    session = int(time.time())
    x_test, y_test = dataset.get_dataset(df=dataset.load_dataset_test())


    scaler = MinMaxScaler()
    x_test = scaler.fit_transform(x_test)
    y_test = utils.label_to_categorical(y_test)
    
    utils.init_scaler(scaler)
    model = utils.get_model(inshape=x_test.shape[1])

    with open('config_training.json', 'w+') as config_training:
        config=config_training.read()        

        try :
            data = json.loads(config)
        except json.JSONDecodeError:
            data={}

        data['num_rounds']=num_rounds
        data['is_resume']=is_resume
        data['session']= session
        data['dataset'] = dataset_name
        data['ctgan'] = enable_ctgan
        json.dump(data,config_training)

    if not (os.path.exists(f'./logs/Session-{session}')):
        os.mkdir(f"./logs/Session-{session}")      

    with open('./logs/Session-{session}/info', 'a') as file:
        file.write('{}')
    # Load last session parameters if they exist
    if not (os.path.exists('./save-weights/fl_sessions')):
    # create fl_sessions directory if first time
        os.mkdir('./save-weights/fl_sessions')

    # initialise sessions list and initial parameters
    sessions = []
    initial_params = None

    # loop through fl_sessions sub-folders and get the list of directories containing the weights 
    for root, dirs, files in os.walk("./save-weights/fl_sessions", topdown = False):
        for name in dirs:
            if name.find('Session')!=-1:
                hist_session = name.strip('Session-')
                sessions.append(hist_session)
               

    if (is_resume and len(sessions)!=0):
        # test if we will start training from the last session weights and
        # if we have at least a session directory
        if os.path.exists(f'./save-weights/fl_sessions/Session-{sessions[-1]}/global_session_{sessions[-1]}_model.npy'):
            # if the latest session directory contains the global model parameters
            initial_parameters = np.load(f"./save-weights/fl_sessions/Session-{sessions[-1]}/global_session_{sessions[-1]}_model.npy", allow_pickle=True)
            # load latest session's global model parameters
            initial_params = initial_parameters[0]
            # model.set_weights(initial_params)

    # Create strategy
    strategy = SaveModelStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        evaluate_fn=get_evaluate_fn(model, x_test, y_test),
        on_fit_config_fn=get_on_fit_config_fn(num_rounds),
        on_evaluate_config_fn=get_on_evaluate_config_fn(num_rounds),
        initial_parameters = initial_params,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    # Add strategy to the blockchain
    strat_added_BC = blockchainService.addStrategy(session,'FedAvg',num_rounds,strategy.__getattribute__('min_available_clients'), dataset.get_dataset_type())

    # Start Flower server
    fl.server.start_server(
        server_address="127.0.0.1:18922",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    for client in strategy.contribution.keys():
        if client != 'total_data_size':
            blockchainService.addContribution(
                _rNo = strategy.contribution[client]['num_rounds_participated'],
                _dataSize= strategy.contribution[client]['data_size'],
                _client_address = strategy.contribution[client]['client_address'],
                _totalDataSize = strategy.contribution['total_data_size'],
                _totalBudget = budget,
                number_of_rounds= num_rounds
            )



 
@app.get('/')
def testFAST():
    return("Hello from server!")
