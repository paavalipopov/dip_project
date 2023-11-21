# pylint: disable=invalid-name
"""Auxilary functions"""

import os
import glob
import shutil

from omegaconf import open_dict, OmegaConf, DictConfig
import pandas as pd

from src.settings import UTCNOW, LOGS_ROOT


def set_project_name(cfg: DictConfig):
    """set project name and project dir based on config"""

    default_prefix = f"{UTCNOW}"

    if cfg.prefix is None:
        prefix = default_prefix
        used_default_prefix = True
    else:
        if len(cfg.prefix) == 0:
            prefix = default_prefix
            used_default_prefix = True
        else:
            # '-'s are reserved for project name parsing
            prefix = cfg.prefix.replace("-", "_")
            used_default_prefix = False

    model_name = cfg.model.name
    if "default_HP" in cfg.model and cfg.model.default_HP:
        if cfg.mode.name != "tune":
            model_name += "_defHP"

    dataset_name = cfg.dataset.name
    if "multiclass" in cfg.dataset and cfg.dataset.multiclass:
        dataset_name += "_mc"
    if "zscore" in cfg.dataset and cfg.dataset.zscore:
        dataset_name += "_zSC"
    if "filter_indices" in cfg.dataset and not cfg.dataset.filter_indices:
        dataset_name += "_allIDC"
    if "tuning_holdout" in cfg.dataset and cfg.dataset.tuning_holdout:
        dataset_name += "_th"

    project_name = f"{prefix}-{cfg.mode.name}-{model_name}-{dataset_name}"

    if "single_HPs" in cfg and cfg.single_HPs:
        project_name += "-single_HPs"
    if "permute" in cfg:
        if cfg.permute != "None":
            project_name += f"-perm_{cfg.permute}"

    project_dir = str(LOGS_ROOT.joinpath(project_name))

    with open_dict(cfg):
        cfg.project_name = project_name
        cfg.project_dir = project_dir
        cfg.used_default_prefix = used_default_prefix


def set_run_name(cfg: DictConfig, outer_k=None, trial=None, inner_k=None):
    """set wandb run name and run directories"""
    if cfg.mode.name == "tune":
        if ("single_HP" in cfg and cfg.single_HP) or (
            "tuning_holdout" in cfg.dataset and cfg.dataset.tuning_holdout
        ):
            # if cfg.single_HP or cfg.dataset.tuning_holdout are True,
            # we are looking for a single optimal set of HPs
            wandb_trial_name = f"trial_{trial:04d}-k_{inner_k:02d}"
            k_dir = f"{cfg.project_dir}"
        else:
            wandb_trial_name = f"k_{outer_k:02d}-trial_{trial:04d}-k_{inner_k:02d}"
            k_dir = f"{cfg.project_dir}/k_{outer_k:02d}"

        trial_dir = f"{k_dir}/trial_{trial:04d}"
        run_dir = f"{trial_dir}/k_{inner_k:02d}"
        with open_dict(cfg):
            cfg.wandb_trial_name = wandb_trial_name
            cfg.k_dir = k_dir
            cfg.trial_dir = trial_dir
            cfg.run_dir = run_dir

    elif cfg.mode.name == "exp":
        wandb_trial_name = f"k_{outer_k:02d}-trial_{trial:04d}"
        k_dir = f"{cfg.project_dir}/k_{outer_k:02d}"
        run_dir = f"{k_dir}/trial_{trial:04d}"
        with open_dict(cfg):
            cfg.wandb_trial_name = wandb_trial_name
            cfg.k_dir = k_dir
            cfg.run_dir = run_dir


def validate_config(cfg: DictConfig):
    """
    Verify the correctness of the provided config.
    Note: This list of checks is not exhaustive, additional checks happen further in the code.
    Note: Some of the checks are repeated further in the code.
    """
    # if model is to use single set of HPs in EXP mode, you must provide path to .yaml or .json file
    if "single_HPs" in cfg and cfg.single_HPs and cfg.mode.name == "exp":
        assert (
            cfg.model_cfg_path is not None
        ), "You must spcify 'model_cfg_path' if single_HPs is set to True"
        assert isinstance(cfg.model_cfg_path, str)
        assert os.path.isfile(
            cfg.model_cfg_path
        ), "Provided path to model config ({cfg.model_cfg_path}) is incorrect"

    # shuffling the time points in the training samples is only allowed for TS input
    if "permute" in cfg:
        assert cfg.permute in ["None", "Single", "Multiple"]
        if cfg.permute != "None" and "data_type" in cfg.model:
            assert (
                cfg.model.data_type == "TS"
            ), "Time permutation is not allowed for non-TS models"

    # if model is specified as not tunable, abort tuning
    # Note: tunable models must have random_HPs(cfg) function defined in their module
    if cfg.mode.name == "tune":
        if "tunable" in cfg.model:
            assert cfg.model.tunable, "Model is specified as not tunable, aborting"

    # if you are using tuning_holdout for your dataset, make sure to provide tuning_split value.
    # in TUNE mode, 1/tuning_split of the dataset will be used for tuning,
    # and the rest will be used in EXP mode
    if "tuning_holdout" in cfg.dataset and cfg.dataset.tuning_holdout:
        assert (
            cfg.dataset.tuning_split is not None
        ), "you must specify 'exp.tuning_split' if \
                 'exp.tuning_holdout' is set to True"
        assert isinstance(cfg.dataset.tuning_split, int)

    # for resuming experiments you must be using a custom prefix
    if "resume" in cfg and cfg.resume:
        assert cfg.prefix is not None


def get_resume_params(cfg: DictConfig) -> DictConfig:
    """find resume point of an interrupted experiment"""

    assert os.path.exists(
        cfg.project_dir
    ), f"No pre-existing project directory ({cfg.project_dir}) is found"

    interrupted_cfg = OmegaConf.load(f"{cfg.project_dir}/general_config.yaml")
    with open_dict(interrupted_cfg):
        interrupted_cfg.resume = True

    if cfg.mode.name == "tune":
        if ("single_HP" in cfg and cfg.single_HP) or (
            "tuning_holdout" in cfg.dataset and cfg.dataset.tuning_holdout
        ):
            starting_k = 0
            search_dir = cfg.project_dir
        else:
            starting_k = max(len(glob.glob(f"{cfg.project_dir}/k_*")) - 1, 0)
            search_dir = f"{cfg.project_dir}/k_{starting_k:02d}"

        try:
            df = pd.read_csv(f"{search_dir}/trial_runs.csv")
            interrupted_trial = len(df)
        except FileNotFoundError:
            interrupted_trial = 0

        interrupted_dir = f"{search_dir}/trial_{interrupted_trial:04d}"

    elif cfg.mode.name == "exp":
        starting_k = max(len(glob.glob(f"{cfg.project_dir}/k_*")) - 1, 0)
        search_dir = f"{cfg.project_dir}/k_{starting_k:02d}"
        try:
            df = pd.read_csv(f"{search_dir}/fold_runs.csv")
            interrupted_trial = len(df)
        except FileNotFoundError:
            interrupted_trial = 0

        interrupted_dir = f"{search_dir}/trial_{interrupted_trial:04d}"

    print(f"Deleting interrupted run logs in '{interrupted_dir}'")
    try:
        shutil.rmtree(interrupted_dir)
    except FileNotFoundError:
        print("Could not delete interrupted run logs - FileNotFoundError")

    if interrupted_trial == interrupted_cfg.mode.n_trials:
        interrupted_trial = 0
        starting_k += 1
    if starting_k == interrupted_cfg.mode.n_splits:
        raise IndexError("The resumed experiment appears to be finished")

    with open_dict(interrupted_cfg):
        interrupted_cfg.resumed_params = {
            "outer_k": starting_k,
            "trial": interrupted_trial,
        }

    return interrupted_cfg
