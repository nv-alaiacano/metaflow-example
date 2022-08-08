import os
import shutil
from metaflow import FlowSpec, Parameter, step
from merlin.schema import Tags, Schema

from typing import Optional
from nvtabular.workflow import Workflow

from merlin.systems.dag.ops.workflow import TransformWorkflow
from merlin.systems.dag.ops.tensorflow import PredictTensorflow
from merlin.systems.dag.ensemble import Ensemble

from merlin.systems.dag.ops.operator import (
    InferenceDataFrame,
    PipelineableInferenceOperator,
)

default_data_path = os.path.expanduser("~/data/alicpp")


class CustomOp(PipelineableInferenceOperator):
    def export(
        self,
        path: str,
        input_schema: Schema,
        output_schema: Schema,
        params: Optional[dict] = None,
        node_id: Optional[int] = None,
        version: int = 1,
    ):
        """
        Export the class object as a config and all related files to the user defined path.

        Parameters
        ----------
        path : str
            Artifact export path
        input_schema : Schema
            A schema with information about the inputs to this operator.
        output_schema : Schema
            A schema with information about the outputs of this operator.
        params : dict, optional
            Parameters dictionary of key, value pairs stored in exported config, by default None.
        node_id : int, optional
            The placement of the node in the graph (starts at 1), by default None.
        version : int, optional
            The version of the model, by default 1.

        Returns
        -------
        Ensemble_config: dict
            The config for the entire ensemble.
        Node_configs: list
            A list of individual configs for each step (operator) in graph.
        """
        ...

    def from_config(cls, config: dict, **kwargs):
        """
        Instantiate a class object given a config.

        Parameters
        ----------
        config : dict
        **kwargs
          contains the following:
            * model_repository: Model repository path
            * model_version: Model version
            * model_name: Model name

        Returns
        -------
            Class object instantiated with config values
        """
        ...

    def transform(self, df: InferenceDataFrame) -> InferenceDataFrame:
        """Transform the dataframe by applying this operator to the set of input columns

        Parameters
        -----------
        df: Dataframe
            A pandas or cudf dataframe that this operator will work on

        Returns
        -------
        DataFrame
            Returns a transformed dataframe for this operator
        """
        ...


class SystemWorkflow(FlowSpec):
    dlrm_input_path = Parameter(
        "dlrm-input-path",
        help="Folder where we will load the DLRM model from.",
        default=os.path.join(default_data_path, "models", "dlrm"),
    )

    nvt_workflow_input_path = Parameter(
        "nvt-workflow-input-path",
        help="Folder where we will load the DLRM model from.",
        default=os.path.join(default_data_path, "nvt_workflow"),
    )

    ensemble_output_path = Parameter(
        "ensemble-output-path",
        help="Folder where we will load the DLRM model from.",
        default=os.path.join(default_data_path, "models", "ensemble"),
    )

    @step
    def start(self):
        """
        This initial step makes sure that the ensemble output path is empty by removing whatever
        might already be there.

        TODO: replace this with a versioned path rather than overwriting.
        """

        if os.path.exists(self.ensemble_output_path):  # type:ignore
            shutil.rmtree(self.ensemble_output_path)  # type:ignore

        self.next(self.generate_ensemble)

    @step
    def generate_ensemble(self):
        """
        This step
        * Loads the NVTabular workflow and the DLRM model
        * Removes any columns from the nvt_workflow schema that is tagged with Tags.TARGET
        * Builds the ensemble
        * Exports the ensemble
        """
        nvt_workflow = Workflow.load(self.nvt_workflow_input_path)

        # Don't forget to remove the target columns from the input schema, or else
        # you'll get a compilation error. The model does not expect these columns.
        label_columns = nvt_workflow.output_schema.select_by_tag(
            Tags.TARGET
        ).column_names
        nvt_workflow.remove_inputs(label_columns)

        serving_operators = (
            nvt_workflow.input_schema.column_names
            >> TransformWorkflow(nvt_workflow)
            >> PredictTensorflow(self.dlrm_input_path)
        )

        ensemble = Ensemble(serving_operators, nvt_workflow.input_schema)

        ensemble.export(self.ensemble_output_path)
        self.next(self.end)

    @step
    def add_custom_op(self):
        pass

    @step
    def end(self):
        """
        Congratulations, it's done. This step will prent the path of the ensemble model.
        """
        print(f"Ensemble generated in {self.ensemble_output_path}")


if __name__ == "__main__":
    SystemWorkflow()
