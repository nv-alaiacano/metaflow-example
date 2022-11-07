# Generating synthetic data for model training

`01-generate-synthetic-data.py` generates synthetic data that matches the columns/distributions of the open source aliccp dataset. This is one of the [synthetic datasets](https://github.com/NVIDIA-Merlin/models/blob/ea40954806b7ffefa3a7afdcd0da4b81331e21ac/merlin/datasets/synthetic.py) available in Merlin Models.

We generate separate parquet files for user featuers, item features, and contextual features. The contextual features file contains columns for the `user_id`, `item_id`, `timestamp` at which the user/item interaction occurred, and any other contextual information about that interaction.
