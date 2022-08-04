This contains a few examples of using metaflow to generate a Merlin pipeline.

The steps expressed are:

* Generate synthetic data and save as parquet files
* Define an NVTabular workflow
* Fit an NVTabular workflow to the data
* Train a DLRM model
* Generate a systems Ensemble with the NVTabular workflow and DLRM model

Some self-enforced rules are:

* All entities (workflows, models, etc) must be serialized to disk between steps
* Run steps in parallel when possible
