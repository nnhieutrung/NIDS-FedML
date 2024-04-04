pragma solidity >=0.5.0 <0.9.0;

contract Federation{
    struct Weight{
        uint dataSize;
        string filePath;
        string fileHash;
    }

    struct GlobalModel{
        string filePath;
        string fileHash;
    }

    struct Strategy{
        string algoName;
        uint numRounds;
        uint numClients;
        string dataset;
    }

    struct CTGAN {
        uint maxRows;
        uint minRows;
        address maxClient;
        mapping(address => uint) clients;
        uint numClients;
    }

    mapping (uint => mapping(uint => mapping(address => Weight))) weights;
    mapping (uint => mapping(uint => GlobalModel)) models;
    mapping (uint => Strategy) strategies;
    mapping (uint => mapping(uint => CTGAN)) ctgans;

    event addStrategyEvent(string _algoName, uint _num_round, uint _num_client, string _dataset);
    event addCTGANFitEvent(uint maxRows, uint minRows);
    event addCTGANDatafakeEvent(string datafake, bool complete);

    // Add & Get Weights
    function addWeight(uint _session, uint _roundNum, uint _dataSize, string memory _filePath, string memory _fileHash) public {
        weights[_session][_roundNum][msg.sender].dataSize = _dataSize;
        weights[_session][_roundNum][msg.sender].filePath = _filePath;
        weights[_session][_roundNum][msg.sender].fileHash = _fileHash;
    }

    function getWeight(uint _session, uint _roundNum) public view returns (uint dataSize_, string memory filePath_, string memory fileHash_){
        dataSize_ = weights[_session][_roundNum][msg.sender].dataSize;
        filePath_ = weights[_session][_roundNum][msg.sender].filePath;
        fileHash_ = weights[_session][_roundNum][msg.sender].fileHash;
        return(dataSize_,filePath_,fileHash_);
    }


    // Add & Get GlobalModel
    function addModel(uint _session, uint _roundNum, string memory _filePath, string memory _fileHash) public {
        models[_session][_roundNum].filePath = _filePath;
        models[_session][_roundNum].fileHash = _fileHash; 
    }

    function getModel(uint _session, uint _roundNum) public view returns (string memory filePath_, string memory fileHash_){
        filePath_ = models[_session][_roundNum].filePath;
        fileHash_ = models[_session][_roundNum].fileHash;
        return (filePath_, fileHash_);
    }


    // Add & Get Strategy
    function addStrategy(uint _session, string memory _algoName, uint _numRounds, uint _numClients, string memory _dataset) public returns(string memory no){
        strategies[_session].algoName = _algoName;
        strategies[_session].numRounds = _numRounds;
        strategies[_session].numClients = _numClients;
        strategies[_session].dataset = _dataset;
        
        emit addStrategyEvent(_algoName, _numRounds, _numClients, _dataset);
        return "Strategy added";
    }

    function getStrategy(uint _session) public view returns (string memory algoName_, uint numRounds_, uint numClients_){
        algoName_ = strategies[_session].algoName;
        numRounds_ = strategies[_session].numRounds;
        numClients_ = strategies[_session].numClients;
        return(algoName_, numRounds_, numClients_);
    }


    // Add & Get CTGAN DataFake
    function addDatasetInfo(uint _session, uint _roundNum, uint _numRows) public {
        if (ctgans[_session][_roundNum].clients[msg.sender] == 0) {
            ctgans[_session][_roundNum].clients[msg.sender] = _numRows;

            if (_numRows > ctgans[_session][_roundNum].maxRows) {
                ctgans[_session][_roundNum].maxRows = _numRows;
                ctgans[_session][_roundNum].maxClient = msg.sender;
            }

            if (ctgans[_session][_roundNum].minRows == 0 || _numRows < ctgans[_session][_roundNum].minRows) {
                ctgans[_session][_roundNum].minRows = _numRows;
            }


            ctgans[_session][_roundNum].numClients++;

            if (strategies[_session].numClients == ctgans[_session][_roundNum].numClients) {
                emit addCTGANFitEvent(ctgans[_session][_roundNum].maxRows, ctgans[_session][_roundNum].minRows);
            }
        }
    }

    function addDatafake(uint _session, uint _roundNum, string memory _datafake, bool _complete) public {
        if (ctgans[_session][_roundNum].maxClient == msg.sender) {
            emit addCTGANDatafakeEvent(_datafake, _complete);
        }
    }
}   