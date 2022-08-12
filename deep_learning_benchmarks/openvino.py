from typing import Callable, List
from pathlib import Path
import logging
import subprocess
import json

import numpy as np
from openvino.runtime import Core


def convert_model(
    saved_onnx_models_path: str,
    input_shape: List[int],
    batch_sizes: List[int],
    output_dir: str,
) -> None:
    logging.info("Converting ONNX model to OpenVINO format")
    saved_onnx_models_path = Path(saved_onnx_models_path)
    for batch_size in batch_sizes:
        subprocess.run(
            [
                "mo",
                "--input_model",
                str(saved_onnx_models_path / f"batch_{batch_size}.onnx"),
                "--output_dir",
                output_dir,
                "--model_name",
                f"batch_{batch_size}",
                "--data_type",
                "FP32",
                "--input_shape",
                json.dumps([batch_size] + input_shape),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def preprocess_input(x: np.ndarray) -> np.ndarray:
    return x


def get_inference_function(
    openvino_models_path: str, batch_sizes: List[int]
) -> Callable[[np.ndarray], np.ndarray]:
    openvino_models_path = Path(openvino_models_path)

    ie = Core()

    models_for_batch_sizes = {}
    for batch_size in batch_sizes:
        model = ie.read_model(
            model=str(openvino_models_path / f"batch_{batch_size}.xml")
        )
        compiled_model = ie.compile_model(model=model, device_name="CPU")
        models_for_batch_sizes[batch_size] = (
            compiled_model,
            next(iter(compiled_model.outputs)),
        )

    def _func(x):
        batch_size = x.shape[0]
        model, output_layer = models_for_batch_sizes[batch_size]
        return model([x])[output_layer]

    return _func
