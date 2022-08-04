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

There are two examples to run. The first mimics the [04-Exporting-ranking-models.ipynb](https://github.com/NVIDIA-Merlin/models/blob/main/examples/04-Exporting-ranking-models.ipynb) example in `models`, and the second mimics the [Serving-Ranking-Models-With-Merlin-Systems.ipynb](https://github.com/NVIDIA-Merlin/systems/blob/main/examples/Serving-Ranking-Models-With-Merlin-Systems.ipynb) in `systems`.

To run them sequentially:

```bash
python examples/01-model_export.py run
python examples/02-system-ensemble.py run
```
