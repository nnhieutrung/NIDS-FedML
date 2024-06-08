# CTGAN Generate fakedata and sync in Client

## Process:
- FOR EACH `label` IN **OUTPUT_LABELS** from the **dataset**:
    1. For each `client`:
        - Get `numRows` = number of rows with `label`.
        - Send `numRows` to the **blockchain**.
    2. Receive `minRows` and `maxRows` from the **blockchain**.
    3. If `maxRows/minRows` < 1.25:
        - **Continue**: Skip this `label`.
    4. If a **client** has `numRows` = `maxRows`, this client will generate fake data:
        - If CTGAN Model does not exist:
            - Fetch every 500 rows per `label`.
            - Use these rows for training the CTGAN model.
        - Calculate `datafake_length = maxRows - minRows`.
        - Initialize `list_datafake` as an empty list.
        - While `len(list_datafake) < datafake_length*2`:
            - Generate datafake of length = datafake_length using the CTGAN model (Which have all `label`).
            - Filter out duplicates and rows that do not have this `label`.
            - Add datafake to list_datafake.
        - Select a random subset of size datafake_length from list_datafake (Which have minimum length of `2*datafake_length`).
        - Convert data to a string.
        - Split the string into `chunks` of length 30,000 characters:
            - Send each `chunk` to the **blockchain**.
    5. If a client has `numRows < maxRows`, this client will receive the datafake:
        - Receive `chunks` from the **blockchain**.
        - Merge the `chunks` into the `datafake` string.
        - Convert the `datafake` string back to a dataset format.
        - Merge the `datafake` into the client's `dataset`.
        - Shuffle the  client's `dataset`.
