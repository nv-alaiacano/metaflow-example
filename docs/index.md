# Example docs.

This is an example of running some Merlin workflows in metaflow.

The demo in `examples/feast` configures a Feast feature store with some synthetic data (in steps 01 and 02) and then trains a model loading the historical data in that feature store (in step 03).

## Stages

### Data generation

`01-generate-synthetic-data.py` generates synthetic data that matches the columns/distributions of the open source aliccp dataset. This is one of the [synthetic datasets](https://github.com/NVIDIA-Merlin/models/blob/ea40954806b7ffefa3a7afdcd0da4b81331e21ac/merlin/datasets/synthetic.py) available in Merlin Models.

We generate separate parquet files for user featuers, item features, and contextual features. The contextual features file contains columns for the `user_id`, `item_id`, `timestamp` at which the user/item interaction occurred, and any other contextual information about that interaction.

### Feature Store ingestion

In `02-ingest-data-into-feast.py` we configure a new feature store using [Feast](http://www.feast.dev).

We are using Feast version 0.19, and are configuring it as a _local_ feature store. The feature store itself is configured in the `feast_feature_store` directory, including the two `FeatureViews`: `UserFeatures` and `ItemFeatures`.

The `02-ingest-data-into-feast.py` script will `apply` the feature store configuration and `materialize` the user- and item-features from the parquet files containing the synthesized data.

### DLRM Model Training

In `03-train-model-from-feature-store.py` we train a DLRM ranking model for predicting user/item interactions. This is a pretty heavy workflow that does a number of things:

First it loads the user/item interaction data + contextual features that we generated in the first step. Using the `(user_id, item_id, timestamp)` triplets, it fetches the appropriate historical user- and item-features from the Feast offline store.

We then define our [NVTabular](http://www.github.com/NVIDIA-Merlin/NVTabular) workflow. All of our features happen to be categorical, so we use the `Categorify` Op on all of them to prepare for training.

Finally, we train a DLRM model using Merlin Models using the `click` field as a binary classification target.

Both the NVTabular workflow and DLRM model are saved to disk for use when serving. It is important that the workflow used to train a model is also used at serving time, so make sure these are somehow associated in your model registry.

### Exporting for training

At this point we have trained a model and are ready to serve it. The assets that we have generated and are required for serving are:

- The Feast feature store containing `UserFeatures` and `ItemFeatures`
- An NVTabular workflow defining how to transform those features
- A DLRM model for predicting the likelihood that the given user will click on the given item.

We now have a lot of flexibility to design how the system will work.

- The request to the model will provide a `user_id` and a list of `item_id`s.
- We will use `QueryFeast` to fetch featuers about this user and these items.
- We will use `UnrollFeatures` to ensure that each record going into the NVTabular workflow has all of the necessary features about the user and item.
- We will use `TransformWorkflow` to transform the data (using the `Categorify` ops that we defined) for prediction.
- We will use `PredictTensorflow` to predict the liklihood that the user will click on each item.
