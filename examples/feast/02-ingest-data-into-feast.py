import subprocess

from metaflow import FlowSpec, Parameter, step


class MaterializeFeastForDemo(FlowSpec):
    """
    This materializes data into the FeatureStore. The expectations are that there is a directory
    containing:

    * A feature_store.yaml file configuring the feature store itself.
    * The parquet data files.
    * Python file(s) defining the FeatureViews to be materialized

    For this demo, see the `feast_feature_store` directory for all of these files. The parquet
    data file are generated with `GenerateFeatureStoreData`

    Flow Parameters:
        FEATURE_STORE_PATH : str - path to the store.
    """

    FEATURE_STORE_PATH = Parameter(
        "feature-store-path",
        help="Path to the local feature store to be materialized.",
        type=str,
        default="feast_feature_store",
    )

    TEARDOWN = Parameter(
        "teardown",
        help="Tear down the existing feature store infrastructure before ingesting new data. "
        + "This is not recommended for iterative data ingestions, but is probably the right way "
        + "to go if you're just experimenting with a static data set.",
        type=bool,
    )

    START_TIME = Parameter(
        "start-time",
        help="start_time parameter for feast materialize commend. Only features created after "
        + "this time will be materialized. Format is YYYY-MM-DDTHH:MM:SS",
        default="2021-07-16T19:20:01",
    )

    END_TIME = Parameter(
        "end-time",
        help="end_time parameter for feast materialize commend. Only features created before "
        + "this time will be materialized. Format is YYYY-MM-DDTHH:MM:SS",
        default="2025-07-16T19:20:01",
    )

    @step
    def start(self):

        self.next(self.teardown_existing)

    @step
    def teardown_existing(self):
        """
        Tears down the existing feature store if --teardown flag was applied. Otherwise, skips
        to the next step.
        """
        if self.TEARDOWN is True:
            print("Removing existing feature store")
            td = subprocess.run(
                ["feast", "-c", str(self.FEATURE_STORE_PATH), "teardown"]
            )
            td.check_returncode()

        else:
            print("Skipping teardown step")
        self.next(self.feast_apply)

    @step
    def feast_apply(self):
        """
        Calls `feast apply` to create the feature views defined in the feature repo directory.
        """
        feast_apply = subprocess.run(
            ["feast", "-c", str(self.FEATURE_STORE_PATH), "apply"]
        )
        feast_apply.check_returncode()
        self.next(self.feast_materialize)

    @step
    def feast_materialize(self):
        """
        Calls `feast materialize` to ingest new data into the store.

        For a more realistic system, you would want to use `feast materialize-incremental`. See
        https://docs.feast.dev/how-to-guides/running-feast-in-production#2.1.-manual-materializations
        for more info.
        """
        feast_materialize = subprocess.run(
            [
                "feast",
                "-c",
                str(self.FEATURE_STORE_PATH),
                "materialize",
                "2021-07-16T19:20:01",
                "2025-07-16T19:20:01",
            ]
        )
        feast_materialize.check_returncode()
        self.next(self.validate)

    @step
    def validate(self):
        print("Validating that it worked.")
        import feast

        fs = feast.FeatureStore(self.FEATURE_STORE_PATH)
        fv = fs.list_feature_views()

        print(f"Found feature views: {[f.name for f in fv]}")

        serv = fs.list_feature_services()
        print(f"Found feature services: {[s.name for s in serv]}")

        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    MaterializeFeastForDemo()
