from ast import Call
from typing import Callable

import torch
import numpy as np


def load_model(model_path: str) -> torch.nn.Module:
    model = torch.load(model_path)
    return model


def preprocess_input(x: np.ndarray) -> torch.Tensor:
    x = torch.Tensor(x)
    return x


def get_inference_function(
    model: torch.nn.Module,
) -> Callable[torch.Tensor, torch.Tensor]:
    def _func(x):
        return model(x)

    return _func
