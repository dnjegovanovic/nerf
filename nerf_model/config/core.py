from pathlib import Path
from typing import Dict
from pydantic import BaseModel
from strictyaml import YAML, load
from yaml.loader import FullLoader
import yaml
import nerf_model

# Project Directories
PACKAGE_ROOT = Path(nerf_model.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"


class AppConfig(BaseModel):
    package_name: str
    save_file: str
    training_sample: int
    testimg_idx: int


class NerfModelConfig(BaseModel):
    data_dir: str
    alpha: float


class StratifiedConfig(BaseModel):
    strf_samp_option: Dict


class NeRFModelTrainConfig(BaseModel):
    data_dir: str
    save_file: str
    strf_samp_option: Dict
    encoder: Dict
    model: Dict
    hierarchical_sampling: Dict
    optimizer: Dict
    training_config: Dict
    early_stoping_config: Dict


class Config(BaseModel):
    """Master config object."""

    # app_config: AppConfig
    # model_config: NerfModelConfig
    # stratified_config: StratifiedConfig
    model_training_config: NeRFModelTrainConfig


def find_config_file() -> Path:
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Path = None) -> YAML:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    with open(cfg_path, "r") as stream:
        try:
            # Converts yaml document to python object
            parsed_config = yaml.load(stream, Loader=FullLoader)
            return parsed_config
        except yaml.YAMLError as e:
            print(e)


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()
        for k, v in parsed_config.items():
            if k == "strf_samp_option":
                parsed_config[k]["n_samples"] = int(parsed_config[k]["n_samples"])
                parsed_config[k]["perturb"] = bool(parsed_config[k]["perturb"])
                parsed_config[k]["inverse_depth"] = bool(
                    parsed_config[k]["inverse_depth"]
                )
                parsed_config[k]["near"] = float(parsed_config[k]["near"])
                parsed_config[k]["far"] = float(parsed_config[k]["far"])
            elif k == "encoder":
                parsed_config[k]["d_input"] = int(parsed_config[k]["d_input"])
                parsed_config[k]["n_freqs"] = int(parsed_config[k]["n_freqs"])
                parsed_config[k]["log_space"] = bool(parsed_config[k]["log_space"])
                parsed_config[k]["use_viewdirs"] = bool(
                    parsed_config[k]["use_viewdirs"]
                )
                parsed_config[k]["n_freqs_views"] = int(
                    parsed_config[k]["n_freqs_views"]
                )
            elif k == "model":
                parsed_config[k]["d_filter"] = int(parsed_config[k]["d_filter"])
                parsed_config[k]["n_layers"] = int(parsed_config[k]["n_layers"])
                parsed_config[k]["skip"] = list(parsed_config[k]["skip"])
                parsed_config[k]["use_fine_model"] = bool(
                    parsed_config[k]["use_fine_model"]
                )
                parsed_config[k]["d_filter_fine"] = int(
                    parsed_config[k]["d_filter_fine"]
                )
                parsed_config[k]["n_layers_fine"] = int(
                    parsed_config[k]["n_layers_fine"]
                )
            elif k == "hierarchical_sampling":
                parsed_config[k]["n_samples_hierarchical"] = int(
                    parsed_config[k]["n_samples_hierarchical"]
                )
                parsed_config[k]["perturb_hierarchical"] = bool(
                    parsed_config[k]["perturb_hierarchical"]
                )
                parsed_config[k]["raw_noise_std"] = float(
                    parsed_config[k]["raw_noise_std"]
                )
                parsed_config[k]["white_bkgd"] = bool(parsed_config[k]["white_bkgd"])
            elif k == "optimizer":
                parsed_config[k]["lr"] = float(parsed_config[k]["lr"])
            elif k == "training_config":
                parsed_config[k]["n_iters"] = int(parsed_config[k]["n_iters"])
                parsed_config[k]["batch_size"] = int(parsed_config[k]["batch_size"])
                parsed_config[k]["one_image_per_step"] = bool(
                    parsed_config[k]["one_image_per_step"]
                )
                parsed_config[k]["chunksize"] = int(parsed_config[k]["chunksize"]) ** 14
                parsed_config[k]["center_crop"] = bool(parsed_config[k]["center_crop"])
                parsed_config[k]["center_crop_iters"] = int(
                    parsed_config[k]["center_crop_iters"]
                )
                parsed_config[k]["display_rate"] = int(parsed_config[k]["display_rate"])
            elif k == "early_stoping_config":
                parsed_config[k]["warmup_iters"] = int(parsed_config[k]["warmup_iters"])
                parsed_config[k]["warmup_min_fitness"] = float(
                    parsed_config[k]["warmup_min_fitness"]
                )
                parsed_config[k]["n_restarts"] = int(parsed_config[k]["n_restarts"])

    _config = Config(
        # app_config=AppConfig(**parsed_config),
        # model_config=NerfModelConfig(**parsed_config),
        # stratified_config=StratifiedConfig(**parsed_config),
        model_training_config=NeRFModelTrainConfig(**parsed_config),
    )

    return _config


config = create_and_validate_config()
