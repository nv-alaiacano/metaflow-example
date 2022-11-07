### Exporting for serving

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
