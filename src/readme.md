# Tutorial

## Setup Requirements
- Install Anacoda
- Run this command
```shell
conda env create --file venv.yml
```
- Install NodeJS

## Setup Ethereum Blockchain :
- Install Ganache: [link](https://archive.trufflesuite.com/ganache/)
- Click **New Workspace** to create WROKSPACE (Only first Time)
- In Tab **WORKSPACE**, Click **ADD PROJECT** and redirect ./Blockchain/truffle-config.js in source
- In Tab **SERVER** Setup port 8545.
- In Tab **ACCOUNTS & KEYS** increase balance (Recommend 1000 ETH) and choose *Autogenrate...*
- (OPTIONAL) In Tab **ADVANCED** config the Logging Path.
- Starting Workspace

## Setup Smart Contract
- Install Truffle by NodeJS Package Manager
```shell
npm install -g truffle
```
- Compile contracts
```shell
cd ./Blockchain
truffle compile
```
- Migrate contracts to network
```shell
cd ./Blockchain
truffle migrate --network development
```

## Setup IPFS
- Signin and Login to [pinata](https://app.pinata.cloud/developers)
- Access to [API-Keys Panel](https://app.pinata.cloud/developers/api-keys)
- Generate new key
- Create file **api_key.json** 
```json
{
    "api_key": "YOUR_PINATA_API_KEY",
    "secret_key": "YOUR_SECRET_KEY",
    "JWT": "YOUR_JWT"
}
```

## Running
- Run server
```shell
uvicorn server:app --reload
```
- Run client
```shell
uvicorn client:app --reload --port PORT_NUMBER
```
Recommend (Client 0: 8001, Client 1: 8002, ...)

- Access API Docs Panel:
    - Server: http://localhost:8000/docs
    - Client: http://localhost:8001/docs, http://localhost:8002/docs

- Init ClientID each Client
    - Client 0: run `GET / Testfast` client_id = 0
    - Client 1: run `GET / Testfast` client_id = 1

- Launch FL: Setup parameters and execute ACTION