# pylint: disable=invalid-name, missing-function-docstring
""" Simple CNN model module """
from random import uniform, randint

import torch
from torch import nn

from omegaconf import OmegaConf, DictConfig


def get_model(cfg: DictConfig, model_cfg: DictConfig):
    return BaselineCNN(model_cfg)


def default_HPs(cfg: DictConfig):
    model_cfg = {
        "lr": 3e-3,
        "input_channels": cfg.dataset.data_info.main.data_shape[1],
        "input_size": cfg.dataset.data_info.main.data_shape[2],
        "output_size": cfg.dataset.data_info.main.n_classes,
    }
    return OmegaConf.create(model_cfg)


def random_HPs(cfg: DictConfig):
    model_cfg = {
        "lr": 10 ** uniform(-4, -3),
        "input_channels": cfg.dataset.data_info.main.data_shape[1],
        "input_size": cfg.dataset.data_info.main.data_shape[2],
        "output_size": cfg.dataset.data_info.main.n_classes,
    }
    return OmegaConf.create(model_cfg)


class BaselineCNN(nn.Module):
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
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.output_figure_size = int(model_cfg.input_size / 2**6)
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * self.output_figure_size**2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, model_cfg.output_size),
        )

    def forward(self, x: torch.Tensor):
        cnn_output = self.cnn(x)

        logits = self.dense(cnn_output)

        return logits
