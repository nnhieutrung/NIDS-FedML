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

    mapping (uint => mapping(uint => mapping(address => Weight))) weights;
    mapping (uint => mapping(uint => GlobalModel)) models;
    mapping (uint => Strategy) strategies;
    event addStrategyEvent(string _algoName, uint _num_round, uint _num_client, string dataset);

    // Add & Get Weights
    function addWeight(uint _session, uint _round_num, uint _dataSize, string memory _filePath, string memory _fileHash) public {
        weights[_session][_round_num][msg.sender].dataSize = _dataSize;
        weights[_session][_round_num][msg.sender].filePath = _filePath;
        weights[_session][_round_num][msg.sender].fileHash = _fileHash;
    }

    function getWeight(uint _session, uint _round_num) public view returns (uint dataSize_, string memory filePath_, string memory fileHash_){
        dataSize_ = weights[_session][_round_num][msg.sender].dataSize;
        filePath_ = weights[_session][_round_num][msg.sender].filePath;
        fileHash_ = weights[_session][_round_num][msg.sender].fileHash;
        return(dataSize_,filePath_,fileHash_);
    }


    // Add & Get GlobalModel
    function addModel(uint _session, uint _round_num, string memory _filePath, string memory _fileHash) public {
        models[_session][_round_num].filePath = _filePath;
        models[_session][_round_num].fileHash = _fileHash; 
    }

    function getModel(uint _session, uint _round_num) public view returns (string memory filePath_, string memory fileHash_){
        filePath_ = models[_session][_round_num].filePath;
        fileHash_ = models[_session][_round_num].fileHash;
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
}