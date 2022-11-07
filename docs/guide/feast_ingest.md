# Ingesting data into Feast

In `02-ingest-data-into-feast.py` we configure a new feature store using [Feast](http://www.feast.dev). [^1]

We are using Feast version 0.19, and are configuring it as a _local_ feature store. The feature store itself is configured in the `feast_feature_store` directory, including the two `FeatureViews`: `UserFeatures` and `ItemFeatures`.

The `02-ingest-data-into-feast.py` script will `apply` the feature store configuration and `materialize` the user- and item-features from the parquet files containing the synthesized data.

[^1]: Merlin Systems currently only supports Feast version 0.19.
