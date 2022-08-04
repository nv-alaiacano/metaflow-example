import os
import shutil
from metaflow import FlowSpec, Parameter, step
from merlin.schema import Tags

from nvtabular.workflow import Workflow

from merlin.systems.dag.ops.workflow import TransformWorkflow
from merlin.systems.dag.ops.tensorflow import PredictTensorflow
from merlin.systems.dag.ensemble import Ensemble


class SystemWorkflow(FlowSpec):
    DATA_FOLDER = Parameter(
        "data-folder",
        help="Folder where we will download/cache data and use it in later steps.",
        default=os.path.expanduser("~/data/alicpp"),
    )

    @step
    def start(self):
        # Identify the path suffixes for the inputs
        self.dlrm_model_path = os.path.join(self.DATA_FOLDER, "models", "dlrm")
        self.nvt_workflow_path = os.path.join(self.DATA_FOLDER, "nvt_workflow")

        # Define path suffix for the output
        self.ensemble_output_path = os.path.join(self.DATA_FOLDER, "ensemble")

        if os.path.exists(self.ensemble_output_path):
            shutil.rmtree(self.ensemble_output_path)

        self.next(self.generate_ensemble)

    @step
    def generate_ensemble(self):

        nvt_workflow = Workflow.load(self.nvt_workflow_path)

        # Don't forget to remove the target columns from the input schema, or else
        # you'll get a compilation error. The model does not expect these columns.
        label_columns = nvt_workflow.output_schema.select_by_tag(
            Tags.TARGET
        ).column_names
        nvt_workflow.remove_inputs(label_columns)

        serving_operators = (
            nvt_workflow.input_schema.column_names
            >> TransformWorkflow(nvt_workflow)
            >> PredictTensorflow(self.dlrm_model_path)
        )

        ensemble = Ensemble(serving_operators, nvt_workflow.input_schema)

        ensemble.export(self.ensemble_output_path)
        self.next(self.end)

    @step
    def end(self):
        print(f">>> Ensemble generated in {self.ensemble_output_path}")


if __name__ == "__main__":
    SystemWorkflow()
