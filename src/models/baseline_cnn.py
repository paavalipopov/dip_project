# pylint: disable=invalid-name, missing-function-docstring
""" MLP model module """
from random import uniform, randint

import torch
from torch import nn

from omegaconf import OmegaConf, DictConfig


def get_model(cfg: DictConfig, model_cfg: DictConfig):
    return BaselineCNN(model_cfg)


def default_HPs(cfg: DictConfig):
    model_cfg = {
        # "dropout": 0.11,
        # "hidden_size": 150,
        # "num_layers": 0,
        # "lr": 0.00027,
        "input_channels": cfg.dataset.data_info.main.data_shape[3],
        "input_size": cfg.dataset.data_info.main.data_shape[1],
        "output_size": cfg.dataset.data_info.main.n_classes,
    }
    return OmegaConf.create(model_cfg)


def random_HPs(cfg: DictConfig):
    model_cfg = {
        # "dropout": uniform(0.1, 0.9),
        # "hidden_size": randint(32, 256),
        # "num_layers": randint(0, 4),
        # "lr": 10 ** uniform(-4, -3),
        "input_channels": cfg.dataset.data_info.main.data_shape[3],
        "input_size": cfg.dataset.data_info.main.data_shape[1],
        "output_size": cfg.dataset.data_info.main.n_classes,
    }
    return OmegaConf.create(model_cfg)


class ResidualBlock(nn.Module):
    """Residual block"""

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x: torch.Tensor):
        return self.block(x) + x


class BaselineCNN(nn.Module):
    """
    MLP model for fMRI data.
    Expected input shape: [batch_size, time_length, input_feature_size].
    Output: [batch_size, n_classes]

    Hyperparameters expected in model_cfg:
        dropout: float
        hidden_size: int
        num_layers: int
    Data info expected in model_cfg:
        input_size: int - input_feature_size
        output_size: int - n_classes
    """

    def __init__(self, model_cfg: DictConfig):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(model_cfg.input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        output_figure_size = model_cfg.input_size / 2**5 * 64
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * output_figure_size**2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, model_cfg.output_size),
        )

    def forward(self, x: torch.Tensor):
        cnn_output = self.cnn(x)

        logits = self.dense(cnn_output)

        return logits
