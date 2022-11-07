### DLRM Model Training

In `03-train-model-from-feature-store.py` we train a DLRM ranking model for predicting user/item interactions. This is a pretty heavy workflow that does a number of things:

First it loads the user/item interaction data + contextual features that we generated in the first step. Using the `(user_id, item_id, timestamp)` triplets, it fetches the appropriate historical user- and item-features from the Feast offline store.

We then define our [NVTabular](http://www.github.com/NVIDIA-Merlin/NVTabular) workflow. All of our features happen to be categorical, so we use the `Categorify` Op on all of them to prepare for training.

Finally, we train a DLRM model using Merlin Models using the `click` field as a binary classification target.

Both the NVTabular workflow and DLRM model are saved to disk for use when serving. It is important that the workflow used to train a model is also used at serving time, so make sure these are somehow associated in your model registry.
