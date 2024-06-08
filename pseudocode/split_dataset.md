# Split dataset in Client
Each client will have their own dataset, and this dataset will be divided in each round.

## Function split_dataset
**Input:**
- `df`: client dataset.
- `round`: current round number.
- `max_round`: total number of rounds

**Output:** `new_df` := 1 a part of a dataset.

1. **if** `round` == `max_round`:
    - **Return** `new_df` = `df`.
2. `new_size` := `round/max_round`.
3. `new_df` := Get a part of a **dataset** from `df` with **size**=`new_size` and **random_state**=`42`.
3. **Return** `new_df`.
