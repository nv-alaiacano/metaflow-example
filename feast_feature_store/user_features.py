import os
from google.protobuf.duration_pb2 import Duration
from feast import Entity, Feature, FeatureView, ValueType
from feast.infra.offline_stores.file_source import FileSource

user_features = FileSource(
    path=os.path.join(os.getcwd(), "data", "user_features.parquet"),
    event_timestamp_column="datetime",
    created_timestamp_column="created",
)

user = Entity(
    name="user_id",
    value_type=ValueType.INT32,
    description="user id",
)

user_features_view = FeatureView(
    name="user_features",
    entities=["user_id"],
    ttl=Duration(seconds=86400 * 7),
    features=[
        Feature(name="user_shops", dtype=ValueType.INT32),
        Feature(name="user_profile", dtype=ValueType.INT32),
        Feature(name="user_group", dtype=ValueType.INT32),
        Feature(name="user_gender", dtype=ValueType.INT32),
        Feature(name="user_age", dtype=ValueType.INT32),
        Feature(name="user_consumption_2", dtype=ValueType.INT32),
        Feature(name="user_is_occupied", dtype=ValueType.INT32),
        Feature(name="user_geography", dtype=ValueType.INT32),
        Feature(name="user_intentions", dtype=ValueType.INT32),
        Feature(name="user_brands", dtype=ValueType.INT32),
        Feature(name="user_categories", dtype=ValueType.INT32),
    ],
    online=True,
    input=user_features,
    tags=dict(),
)
