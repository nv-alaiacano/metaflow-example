import shutil
import os
import nvtabular as nvt
from pathlib import Path
from metaflow import FlowSpec, Parameter, step
from merlin.datasets.synthetic import generate_data
from merlin.io.dataset import Dataset
from merlin.schema import Tags
import merlin.models.tf as mm
import tensorflow as tf


class DLRMModelExportFlow(FlowSpec):

    DATA_FOLDER = Parameter(
        "data-folder",
        help="Folder where we will download/cache data and use it in later steps.",
        default=os.path.expanduser("~/data/alicpp"),
    )

    FEATURE_STORE_FOLDER = Parameter(
        "feature-store-path", default=os.path.join(os.getcwd(), "feast_feature_store")
    )

    INTERACTIONS_FILE = Parameter(
        "interactions",
        help="Path to a parquet file that contains user/item interactions",
        type=str,
        default=os.path.join(
            os.getcwd(), "feast_feature_store", "data", "context_features.parquet"
        ),
    )

    NUM_ROWS = Parameter(
        "num-rows", help="Number of rows to generate", default=1_000_000, type=int
    )

    SYNTHETIC_DATA = Parameter(
        "synthetic-data", help="Use/generate synthetic data", default=True, type=bool
    )

    BATCH_SIZE = Parameter(
        "batch-size", help="Training batch size", default=512, type=int
    )

    @step
    def start(self):
        """
        Initializes the flow by ensuring that DATA_FOLDER exists, then kicks off the next two tasks
        in parallel: producing the training data, and defining the NVTabular workflow
        """
        print(f"Ensuring that the DATA_FOLDER {self.DATA_FOLDER} exists")
        if not Path(self.DATA_FOLDER).exists():
            Path(self.DATA_FOLDER).mkdir(parents=True, exist_ok=True)
        self.next(self.load_user_item_interactions, self.generate_transform_workflow)

    @step
    def load_user_item_interactions(self):
        """
        Given a list of user/item interactions, we fetch the associated features for the
        interactions from Feast.
        """
        interactions_df = Dataset(self.INTERACTIONS_FILE)
        interactions_df = interactions_df.to_ddf().compute()

        # interactions_df.rename(columns={"datetime": "event_timestamp"}, inplace=True)

        test_split_index = int(interactions_df.shape[0] * 0.7)
        self.train_interactions_df = interactions_df[:test_split_index]
        self.valid_interactions_df = interactions_df[test_split_index:]
        self.next(self.fetch_historical_data_from_feast)

    @step
    def fetch_historical_data_from_feast(self):
        """
        This step fetches training data from feast.
        """
        import feast

        fs = feast.FeatureStore(self.FEATURE_STORE_FOLDER)

        all_features_to_fetch = []
        for view in fs.list_feature_views():
            for feature in view.features:
                all_features_to_fetch.append(f"{view.name}:{feature.name}")

        train_output_path = os.path.join(self.DATA_FOLDER, "train", "train.parquet")
        valid_output_path = os.path.join(self.DATA_FOLDER, "valid", "valid.parquet")
        for p in [train_output_path, valid_output_path]:
            if os.path.exists(p):
                shutil.rmtree(p)

        fs.get_historical_features(
            self.train_interactions_df, all_features_to_fetch
        ).to_df().to_parquet(train_output_path)

        fs.get_historical_features(
            self.valid_interactions_df, all_features_to_fetch
        ).to_df().to_parquet(valid_output_path)

        self.next(self.transform_data)

    @step
    def generate_transform_workflow(self):
        """
        This step generates an NVTabular workflow for feature engineering. It does not actually
        fit the workflow to data yet.

        Separating the workflow definition from fitting will allow us to re-fit the same workflow
        artifact on new daily partitions of data.
        """
        self.nvt_workflow_definition_path = os.path.join(
            self.DATA_FOLDER, "nvt_workflow_definition"
        )

        # clear out the existing workflow definition.
        # TODO: this path should be paramaterized with a version rather than re-writing every time
        if os.path.exists(self.nvt_workflow_definition_path):
            shutil.rmtree(self.nvt_workflow_definition_path)

        from nvtabular.ops import (
            Categorify,
            AddMetadata,
            TagAsItemFeatures,
            TagAsUserFeatures,
            TagAsItemID,
            TagAsUserID,
        )

        user_id = ["user_id"] >> Categorify() >> TagAsUserID()
        item_id = ["item_id"] >> Categorify() >> TagAsItemID()
        targets = ["click"] >> AddMetadata(tags=[Tags.BINARY_CLASSIFICATION, "target"])

        item_features = (
            ["item_category", "item_shop", "item_brand"]
            >> Categorify()
            >> TagAsItemFeatures()
        )

        user_features = (
            [
                "user_shops",
                "user_profile",
                "user_group",
                "user_gender",
                "user_age",
                "user_consumption_2",
                "user_is_occupied",
                "user_geography",
                "user_intentions",
                "user_brands",
                "user_categories",
            ]
            >> Categorify()
            >> TagAsUserFeatures()
        )

        outputs = user_id + item_id + item_features + user_features + targets

        workflow = nvt.Workflow(outputs)

        workflow.save(self.nvt_workflow_definition_path)
        self.next(self.transform_data)

    @step
    def transform_data(self, inputs):
        """
        This step takes the NVTabular workflow and fits it to the training data.

        It then applies that fit workflow to both the training and validation data and stores those
        datasets for training in the next step.
        """
        self.nvt_workflow_path: Path = Path(self.DATA_FOLDER) / "nvt_workflow"
        processed_data_path: Path = Path(self.DATA_FOLDER) / "processed"
        self.processed_train_data = str(processed_data_path / "train")
        self.processed_valid_data = str(processed_data_path / "valid")

        # clear out the existing processed data.
        if processed_data_path.exists():
            shutil.rmtree(processed_data_path)

        processed_data_path.mkdir()

        workflow = nvt.Workflow.load(
            inputs.generate_transform_workflow.nvt_workflow_definition_path
        )

        # Load train/valid data
        train_df = Dataset(os.path.join(self.DATA_FOLDER, "train", "*.parquet"))
        valid_df = Dataset(os.path.join(self.DATA_FOLDER, "valid", "*.parquet"))

        # Fit the workflow to the training data
        workflow.fit(train_df)

        # save the train/validation data
        workflow.transform(train_df).to_parquet(output_path=self.processed_train_data)
        workflow.transform(valid_df).to_parquet(output_path=self.processed_valid_data)

        # Write the fit workflow, which will have schemas etc.
        workflow.save(str(self.nvt_workflow_path))
        self.next(self.train_dlrm)

    @step
    def train_dlrm(self):
        """
        This step trains a DLRM model using the train/valid data splits from the previous step.
        """
        self.model_output_path = os.path.join(self.DATA_FOLDER, "models", "dlrm")

        # Remove any already-stored models
        # TODO: enable cacheing?
        if os.path.exists(self.model_output_path):
            shutil.rmtree(self.model_output_path)

        train = Dataset(os.path.join(self.processed_train_data, "*.parquet"))
        valid = Dataset(os.path.join(self.processed_valid_data, "*.parquet"))

        schema = train.schema
        target_column = schema.select_by_tag(Tags.TARGET).column_names[0]
        model = mm.DLRMModel(
            schema,
            embedding_dim=64,
            bottom_block=mm.MLPBlock([128, 64]),
            top_block=mm.MLPBlock([128, 64, 32]),
            prediction_tasks=mm.BinaryClassificationTask(target_column),
        )

        model.compile("adam", run_eagerly=False, metrics=[tf.keras.metrics.AUC()])
        model.fit(train, validation_data=valid, batch_size=self.BATCH_SIZE)

        model.save(self.model_output_path)

        self.next(self.end)

    @step
    def end(self):
        """
        Congratulations, it's done. This step will simply print the output path.
        """
        print(f">>> DLRM model exported to {self.model_output_path}")
        pass


if __name__ == "__main__":
    DLRMModelExportFlow()
