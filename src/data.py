# pylint: disable=invalid-name, line-too-long
"""Functions for extracting dataset features and labels"""
from importlib import import_module

import numpy as np
from scipy import stats
from sklearn.model_selection import StratifiedKFold

from omegaconf import OmegaConf, DictConfig, open_dict


def data_factory(cfg: DictConfig):
    """
    Model-agnostic data factory.
    1. Loads 'cfg.dataset.name' dataset (requires src.datasets.{cfg.dataset.name}.load_data(cfg) to be defined)
        Loaded data is expected to be features([n_samples, time_length, feature_size]), labels([n_samples])
        Otherwise you need to define custom processor
    2. Selects tuning or experiment portion if cfg.dataset.tuning_holdout is True
    3. Processes the data in common_processor, or some custom processor if
        cfg.dataset.custom_processor is True and src.datasets.{cfg.dataset.name}.get_processor(data, cfg) is defined
    4. Save data_info returned by processor in cfg.dataset.data_info, and return processed data

    Processed data is a dictionary with
    {
        "main": cfg.dataset.name dataset,
    }
    data_info is a dictionary with
    {
        "main": main dataset info (depends on the processor),
    }
    """
    # load dataset
    try:
        dataset_module = import_module(f"src.datasets.{cfg.dataset.name}")
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"No module named '{cfg.dataset.name}' \
                                  found in 'src.datasets'. Check if dataset name \
                                  in config file and its module name are the same"
        ) from e

    try:
        data, labels = dataset_module.load_data(cfg)
    except AttributeError as e:
        raise AttributeError(
            f"'src.datasets.{cfg.dataset.name}' has no function\
                             'load_data'. Is the function misnamed/not defined?"
        ) from e

    # process data
    if "custom_processor" not in cfg.dataset or not cfg.dataset.custom_processor:
        processor = common_processor
    else:
        try:
            dataset_module = import_module(f"src.datasets.{cfg.dataset.name}")
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"No module named '{cfg.dataset.name}' \
                                    found in 'src.datasets'. Check if dataset name \
                                    in config file and its module name are the same"
            ) from e

        try:
            get_processor = dataset_module.get_processor
        except AttributeError as e:
            raise AttributeError(
                f"'src.datasets.{cfg.dataset.name}' has no function\
                                'get_processor'. Is the function misnamed/not defined?"
            ) from e

        processor = get_processor()

    data = {}
    data_info = {}
    data["main"], data_info["main"] = processor(cfg, (data, labels))

    with open_dict(cfg):
        cfg.dataset.data_info = data_info

    return data


def common_processor(cfg: DictConfig, raw_data):
    """
    Return processed data and data_info based on config

    Returns (data, data_info) tuple.
    Data is a dict with
    {
        "data": data,
        "labels": labels
    }

    data_info is a DictConfig with
    {
        "data_shape": shape of the images data,
        "n_classes": n_classes,
    }
    )
    """

    data, labels = raw_data
    n_classes = np.unique(labels).shape[0]

    data_info = OmegaConf.create(
        {
            "data_shape": data_shape,
            "n_classes": n_classes,
        }
    )

    return data, data_info


def data_postfactory(cfg: DictConfig, model_cfg: DictConfig, original_data):
    """
    Post-process the raw dataset according to model_cfg if cfg.model.require_data_postproc is True
    """
    if "require_data_postproc" not in cfg.model or not cfg.model.require_data_postproc:
        data = original_data
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
            data_postproc = model_module.data_postproc
        except AttributeError as e:
            raise AttributeError(
                f"'src.models.{cfg.model.name}' has no function\
                                'data_postproc'. Is the function misnamed/not defined?"
            ) from e

        data = data_postproc(cfg, model_cfg, original_data)

    return data
