import os
import shutil
from metaflow import FlowSpec, Parameter, step
from merlin.schema import Tags

from nvtabular.workflow import Workflow

from merlin.systems.dag.ops.workflow import TransformWorkflow
from merlin.systems.dag.ops.tensorflow import PredictTensorflow
from merlin.systems.dag.ensemble import Ensemble

default_data_path = os.path.expanduser("~/data/alicpp")


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

        if os.path.exists(self.ensemble_output_path):
            shutil.rmtree(self.ensemble_output_path)

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
    def end(self):
        """
        Congratulations, it's done. This step will prent the path of the ensemble model.
        """
        print(f">>> Ensemble generated in {self.ensemble_output_path}")


if __name__ == "__main__":
    SystemWorkflow()
