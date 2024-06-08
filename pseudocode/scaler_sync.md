# Scaler Synchronization
The mechanism for synchronizing the MinMaxScaler between the client and server.



## In Node Server
### Function set_scaler
**Input:**
- `dataset`: Server dataset.

**Variables:**
- `config`: A configuration dictionary that the Flower Server will send to the Flower Client each round.

**Output:** `new_config`: The new config has added a variable `scaler_data` to be sent to the client.

1. Initialize `scaler` as a **MinMaxScaler** object to scale the dataset between `0` and `1`.
2. Fit `scaler` to the `dataset` provided by the server.
3. Retrieve `scaler_data`, which consists of `data_min` and `data_max` for each feature of the dataset.
4. Add `scaler_data` to `config["scaler_data"]`.
5. **Return** `config`.

## In Node Client
### Function get_scaler
**Input:**
- `dataset`: Client dataset.
- `config`: A configuration dictionary received from the Flower Server each round.

**Output:**
- `scaled_dataset`: A scaled dataset.

1. Initialize `scaler` as a **MinMaxScaler** object.
2. Retrieve `scaler_data` from `config["scaler_data"]`.
3. Fit `scaler` using `scaler_data`.
4. `scaled_dataset` = tranform a `dataset` with the fitted scaler.
5. **Return** `scaled_dataset`.

