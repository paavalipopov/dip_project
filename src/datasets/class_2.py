# pylint: disable=too-many-function-args, invalid-name
""" trimmed 2-class dataset """
import numpy as np

from omegaconf import DictConfig

from src.settings import DATA_ROOT


def load_data(
    cfg: DictConfig,
):

    class_1 = np.load(f"{DATA_ROOT}/np_data/512x512/train/class_1.npz")
    class_2 = np.load(f"{DATA_ROOT}/np_data/512x512/train/class_2.npz")

    data = np.stack((class_1["data"], class_2["data"]), axis=0)
    labels = np.stack((class_1["labels"], class_2["labels"]), axis=0) - 1

    return data, labels
