import subprocess
import os
import shutil
from pathlib import Path
import numpy as np
import datetime
from metaflow import FlowSpec, Parameter, step
from merlin.schema import Tags
from merlin.io.dataset import Dataset


class GenerateFeatureStoreData(FlowSpec):
    """
    Generates fake alicpp data, sets up a Feature Store, and calls apply/materialize to get
    data into the store.
    """

    NUM_ROWS = Parameter(
        "num-rows",
        default=10_000,
        help="Number of fake user/item interactions to generate",
        type=int,
    )

    DATA_DIR = Parameter(
        "data-dir",
        help="Location where we will store the synthetic data. This path will be "
        + "used later in the `feast apply` step.",
        type=str,
    )

    OVERWRITE_EXISTING = Parameter(
        "overwrite",
        type=bool,
        help="If true, remove any data in DATA_DIR and re-write",
    )

    @step
    def start(self):
        from merlin.datasets.synthetic import generate_data
        import pandas as pd

        data_dir_path = Path(self.DATA_DIR)  # type:ignore

        if data_dir_path.exists() and self.OVERWRITE_EXISTING is False:
            raise ValueError(
                f"Data already exists in {self.DATA_DIR}. Add --overwrite flag to overwrite it"
            )
        else:
            if data_dir_path.exists():
                shutil.rmtree(str(data_dir_path))
            data_dir_path.mkdir(parents=True)

        raw_data: Dataset = generate_data("aliccp-raw", self.NUM_ROWS)  # type:ignore
        raw_data_df: pd.DataFrame = raw_data.to_ddf().compute()

        # There is no datetime column generated with these, we we'll make random ones!
        # The rules are:
        # Start one week ago, and generate random timestamps over the course of 7 days.
        secs_in_a_week = 60 * 60 * 24 * 7
        raw_data_df["datetime"] = [
            datetime.datetime.now()
            - datetime.timedelta(days=7)
            + datetime.timedelta(seconds=int(rsec))
            for rsec in np.random.uniform(0, secs_in_a_week, raw_data_df.shape[0])
        ]
        raw_data_df["datetime"] = raw_data_df["datetime"].astype("datetime64[ns]")
        raw_data_df["created"] = raw_data_df["datetime"]

        # USER FEATURES
        user_cols = raw_data.schema.select_by_tag(tags=Tags.USER).column_names + [
            "datetime",
            "created",
        ]

        user_df = raw_data_df[user_cols]
        user_df.to_parquet(
            os.path.join(self.DATA_DIR, "user_features.parquet")  # type:ignore
        )
        ucols = "\n".join(["  " + c for c in list(user_df.columns)])
        print(f"Wrote user features:\n {ucols}")
        print(f"Wrote {user_df.shape[0]} rows of user features.")

        # ITEM FEATURES
        item_cols = raw_data.schema.select_by_tag(tags=Tags.ITEM).column_names + [
            "datetime",
            "created",
        ]
        item_df = raw_data_df[item_cols]
        item_df.to_parquet(
            os.path.join(self.DATA_DIR, "item_features.parquet")  # type:ignore
        )
        icols = "\n".join(["  " + c for c in list(item_df.columns)])
        print(f"Wrote item features:\n {icols}")
        print(f"Wrote {item_df.shape[0]} rows of item features.")

        # CONTEXT FEATURES Stored in parquet for fetching historical data later.
        # We want: user_id, item_id, any Tags.CONTEXT features, and we'll generate a fake timestamp.
        context_cols = raw_data.schema.select_by_tag(
            tags=[Tags.USER_ID, Tags.ITEM_ID, Tags.CONTEXT]
        ).column_names + ["datetime", "created"]
        context_df = raw_data_df[context_cols]
        context_df.rename(columns={"datetime": "event_timestamp"}, inplace=True)
        context_df.to_parquet(
            os.path.join(self.DATA_DIR, "context_features.parquet")  # type:ignore
        )
        ccols = "\n".join(["  " + c for c in list(context_df.columns)])
        print(f"Wrote context features:\n {ccols}")
        print(f"Wrote {context_df.shape[0]} rows of context features.")

        # TODO: we are skipping the user-item features. It's not super clear what they are.
        self.next(self.end)

    @step
    def end(self):
        print(
            f"Data created in {self.DATA_DIR}. The next step is to call feast apply & feast materialize."
        )
        pass


if __name__ == "__main__":
    GenerateFeatureStoreData()
