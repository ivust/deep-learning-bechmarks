from typing import Callable, List
from pathlib import Path
import logging

import torch
import numpy as np


def load_model(model_path: str) -> torch.nn.Module:
    logging.info("Loading PyTorch model")
    model = torch.load(model_path)
    return model


def convert_to_onnx(
    model: torch.nn.Module,
    save_path: str,
    input_shape: List[int],
    batch_sizes: List[int],
) -> None:
    logging.info("Converting PyTorch model to ONNX")
    save_path = Path(save_path)
    save_path.mkdir()
    for batch_size in batch_sizes:
        input_batch = torch.Tensor(np.random.randn(batch_size, *input_shape))
        torch.onnx.export(
            model, input_batch, str(save_path / f"batch_{batch_size}.onnx")
        )


def preprocess_input(x: np.ndarray) -> torch.Tensor:
    x = torch.Tensor(x)
    return x


def get_inference_function(
    model: torch.nn.Module,
) -> Callable[[torch.Tensor], torch.Tensor]:
    def _func(x):
        return model(x)

    return _func
