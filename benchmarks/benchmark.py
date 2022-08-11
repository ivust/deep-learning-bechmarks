import json
import click

import pandas as pd
from .pytorch import (
    load_model as pytorch_load_model,
    get_inference_function as pytorch_get_inference_function,
    preprocess_input as pytorch_preprocess_input,
)
from .evaluate import evaluate, analyze_results


@click.command()
@click.option("--pytorch-model-path", type=str, help="Path to the PyTorch model")
@click.option(
    "--openvino", is_flag=True, help="Whether to include OpenVINO optimization"
)
@click.option("--input-shape", type=str, help="Shape of the model input")
def main(pytorch_model_path, openvino, input_shape):
    input_shape = json.loads(input_shape)
    model = pytorch_load_model(pytorch_model_path)
    pytorch_results = evaluate(
        pytorch_get_inference_function(model), pytorch_preprocess_input, input_shape
    )

    combined_results = {"PyTorch": pytorch_results}
    results_df = analyze_results(combined_results)

    # print(results_df.round(2).to_string())
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(results_df.round(2))


if __name__ == "__main__":
    main()
