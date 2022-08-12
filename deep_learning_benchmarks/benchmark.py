import json
import click
import shutil
import logging

import pandas as pd
from .pytorch import (
    load_model as pytorch_load_model,
    preprocess_input as pytorch_preprocess_input,
    get_inference_function as pytorch_get_inference_function,
    convert_to_onnx as pytorch_convert_to_onnx,
)
from .openvino import (
    convert_model as openvino_convert_model,
    preprocess_input as openvino_preprocess_input,
    get_inference_function as openvino_get_inference_function,
)
from .evaluate import evaluate, analyze_results


@click.command()
@click.option("--pytorch-model-path", type=str, help="Path to the PyTorch model")
@click.option(
    "--openvino", is_flag=True, help="Whether to include OpenVINO optimization"
)
@click.option("--input-shape", type=str, help="Shape of the model input")
@click.option(
    "--batch-sizes", type=str, help="Batch sizes to consider for throughput estimation"
)
@click.option(
    "--num-trials", type=int, default=10, help="Number of trials for each batch size"
)
@click.option(
    "--warmup", type=int, default=3, help="Number of initial trials which are discarded"
)
def main(pytorch_model_path, openvino, input_shape, batch_sizes, num_trials, warmup):
    combined_results = {}

    logging.basicConfig(level=logging.INFO)
    logging.info("Start PyTorch model benchmarking")
    model = pytorch_load_model(pytorch_model_path)
    pytorch_results = evaluate(
        inference_function=pytorch_get_inference_function(model),
        preprocessing_function=pytorch_preprocess_input,
        input_shape=json.loads(input_shape),
        batch_sizes=json.loads(batch_sizes),
        warmup=warmup,
        num_trials=num_trials,
    )
    combined_results["PyTorch"] = pytorch_results

    if openvino:
        logging.info("Start OpenVINO model benchmarking")
        pytorch_convert_to_onnx(
            model=model,
            save_path="model_onnx",
            input_shape=json.loads(input_shape),
            batch_sizes=json.loads(batch_sizes),
        )
        openvino_convert_model(
            saved_onnx_models_path="model_onnx",
            input_shape=json.loads(input_shape),
            batch_sizes=json.loads(batch_sizes),
            output_dir="model_openvino",
        )
        openvino_results = evaluate(
            inference_function=openvino_get_inference_function(
                "model_openvino", json.loads(batch_sizes)
            ),
            preprocessing_function=openvino_preprocess_input,
            input_shape=json.loads(input_shape),
            batch_sizes=json.loads(batch_sizes),
            warmup=warmup,
            num_trials=num_trials,
        )
        combined_results["OpenVINO"] = openvino_results

        shutil.rmtree("model_onnx")
        shutil.rmtree("model_openvino")

    results_df = analyze_results(combined_results)
    print(results_df.round(2).to_string())


if __name__ == "__main__":
    main()
