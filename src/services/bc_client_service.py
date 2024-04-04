from web3 import Web3
import json
import time
import sys

from threading import Thread
from config import *

w3 = Web3(Web3.HTTPProvider(ETH_SERVER))

contributionSC = open('./blockchain/build/contracts/Contribution.json')
contributionData = json.load(contributionSC)
contributionAbi = contributionData['abi']
addressContribution = contributionData['networks']['5777']['address']
contribution_contract_instance = w3.eth.contract(address=addressContribution, abi=contributionAbi)


federationSC = open('./blockchain/build/contracts/Federation.json')
federationData = json.load(federationSC)
federationAbi = federationData['abi']
addressFederation = federationData['networks']['5777']['address']
federation_contract_instance = w3.eth.contract(address=addressFederation, abi=federationAbi)

ctgan_sync_filter = federation_contract_instance.events.addCTGANFitEvent.create_filter(fromBlock='latest')
ctgan_datafake_filter = federation_contract_instance.events.addCTGANDatafakeEvent.create_filter(fromBlock='latest')

class BlockchainService():
    def addWeight(self, _session: int, _roundNum: int, _dataSize: int, _filePath: str, _fileHash: str, client_id: int):
        client_address = w3.eth.accounts[client_id+1]
        federation_contract_instance.functions.addWeight(_session, _roundNum, _dataSize, _filePath, _fileHash).transact({'from': client_address})
        result = federation_contract_instance.functions.getWeight(_session,_roundNum).call()
        return result

    def getAddress(self, client_id:int):
        return w3.eth.accounts[client_id+1]

    def getContributions(client_address):
        roundNumbers = contribution_contract_instance.functions.get_rNos().call()
        contributions = []
        for rNo in roundNumbers:
            contribution = contribution_contract_instance.functions.getContribution(client_address, rNo).call()
            if contribution[1]>0:
                contrib_dic={}
                # Bool: Work status
                contrib_dic[0]= contribution[0]
                # Uint: Data size
                contrib_dic[1]= contribution[1]
                # Uint: Account Balance
                contrib_dic[2]= contribution[2]
                # Uint: Number of Round
                contrib_dic[3]= rNo
                contributions.append(contrib_dic)
        return contributions
    

    def getCTGANMaxRows(self, _session: int, _roundNum: int, _numRows: int, client_id: int):
        client_address = w3.eth.accounts[client_id+1]
        federation_contract_instance.functions.addDatasetInfo(_session, _roundNum, _numRows).transact({'from': client_address})
        while True:
            events = ctgan_sync_filter.get_new_entries()
            if len(events) > 0:
                print(events[-1])
                return events[-1].args.maxRows, events[-1].args.minRows
            time.sleep(1)

    def sendCTGANDatafake(self, _session: int, _roundNum: int, _datafake: str, _complete: bool, client_id: int):
        client_address = w3.eth.accounts[client_id+1]
        federation_contract_instance.functions.addDatafake(_session, _roundNum, _datafake, _complete).transact({'from': client_address})
        
    def getCTGANDatafake(self, _session: int, _roundNum: int):
        datafake = ""
        while True:
            for event in ctgan_datafake_filter.get_new_entries():
                datafake = datafake + event.args.datafake

                sys.stdout.write('\r')
                sys.stdout.write("Receiving %d bytes of datafake" % (len(datafake)))
                sys.stdout.flush()
                
                if (event.args.complete):
                    return datafake
            time.sleep(0.01)


def handle_event(event):
    print(event)
    from client import FLlaunch
    FLlaunch = FLlaunch()
    FLlaunch.start(event.args._dataset)

def log_loop(event_filter, poll_interval):
    print(event_filter.get_new_entries())
    while True:
        for event in event_filter.get_new_entries():
            handle_event(event)
        time.sleep(poll_interval)


def main():
    block_filter = federation_contract_instance.events.addStrategyEvent.create_filter(fromBlock='latest')
    worker = Thread(target=log_loop, args=(block_filter, 5), daemon=True)
    worker.start()
        
main()



