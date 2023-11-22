# pylint: disable=no-member, invalid-name, too-many-locals, too-many-arguments, consider-using-dict-items
""" Scripts for creating dataloaders """
from importlib import import_module
from copy import deepcopy

from numpy.random import default_rng

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import torch
from torch.utils.data import DataLoader, TensorDataset


def dataloader_factory(cfg, data, k, trial=None):
    """Return dataloader according to the used model"""
    if "custom_dataloader" not in cfg.model or not cfg.model.custom_dataloader:
        dataloader = common_dataloader(cfg, data, k, trial)
    else:
        try:
            model_module = import_module(f"src.models.{cfg.model.name}")
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"No module named '{cfg.model.name}' \
                                    found in 'src.models'. Check if model name \
                                    in config file and its module name are the same"
            ) from e

        try:
            get_dataloader = model_module.get_dataloader
        except AttributeError as e:
            raise AttributeError(
                f"'src.models.{cfg.model.name}' has no function\
                                'get_dataloader'. Is the function misnamed/not defined?"
            ) from e

        dataloader = get_dataloader(cfg, data, k, trial)

    return dataloader


def common_dataloader(cfg, original_data, k, trial=None):
    """
    Return common dataloaders
    dataloaders are a dictionary of
    {
        "train": dataloader,
        "valid": dataloader,
        "test": dataloader,
    }
    Expects the data produced by src.data.data_factory() with common_processor:
        {
        "main":
            {
            "data": data,
            "labels": data,
            },
        }

    Output dataloaders return tuples with ("data", "labels") data order
    """
    data = original_data
    split_data = {"train": {}, "valid": {}, "test": {}}

    # train/test split
    split_data["train"], split_data["test"] = cross_validation_split(
        data["main"], cfg.mode.n_splits, k
    )

    # train/val split
    splitter = StratifiedShuffleSplit(
        n_splits=cfg.mode.n_trials,
        test_size=split_data["train"]["labels"].shape[0] // cfg.mode.n_splits,
        random_state=42,
    )
    tr_val_splits = list(
        splitter.split(split_data["train"]["labels"], split_data["train"]["labels"])
    )
    train_index, val_index = (
        tr_val_splits[0] if cfg.mode.name == "tune" else tr_val_splits[trial]
    )
    for key in split_data["train"]:
        split_data["train"][key], split_data["valid"][key] = (
            split_data["train"][key][train_index],
            split_data["train"][key][val_index],
        )

    # create dataloaders
    dataloaders = {}
    for key in split_data:
        for data_key in split_data[key]:
            if data_key == "labels":
                split_data[key][data_key] = torch.tensor(
                    split_data[key][data_key], dtype=torch.int64
                )
            else:
                split_data[key][data_key] = torch.tensor(
                    split_data[key][data_key], dtype=torch.float32
                )

        dataloaders[key] = DataLoader(
            TensorDataset(split_data[key]["data"], split_data[key]["labels"]),
            batch_size=cfg.mode.batch_size,
            num_workers=0,
            shuffle=key == "train",
        )

    return dataloaders


def cross_validation_split(data, n_splits, k):
    """
    Split data into train and test data using StratifiedKFold.
    Input data should be dict
    {
        "single_or_multiple_keys": data,
        "labels": data (required),
    }
    Output train and test data inherit the same keys
    """
    train_data = {}
    test_data = {}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    CV_folds = list(skf.split(data["labels"], data["labels"]))
    train_index, test_index = CV_folds[k]
    for key in data:
        train_data[key], test_data[key] = (
            data[key][train_index],
            data[key][test_index],
        )

    return train_data, test_data
