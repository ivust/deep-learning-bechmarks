import json
import click
import pytorch, evaluate


@click.command()
@click.option("--pytorch-model-path", type=str, help="Path to the PyTorch model")
@click.option(
    "--openvino", is_flag=True, help="Whether to include OpenVINO optimization"
)
@click.option("--input-shape", type=str, help="Shape of the model input")
def main(pytorch_model_path, openvino, input_shape):
    input_shape = json.loads(input_shape)
    model = pytorch.load_model(pytorch_model_path)
    pytorch_results = evaluate.evaluate(
        pytorch.get_inference_function(model), pytorch.preprocess_input, input_shape
    )

    combined_results = {"PyTorch": pytorch_results}
    results_df = evaluate.analyze_results(combined_results)

    print(results_df.to_string())
