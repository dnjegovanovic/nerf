from pathlib import Path

from pydantic import BaseModel
from strictyaml import YAML, load

import nerf_model

# Project Directories
PACKAGE_ROOT = Path(nerf_model.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"

class AppConfig(BaseModel):
    package_name: str
    save_file: str

class NerfModelConfig(BaseModel):
    data_dir: str
    alpha: float

class Config(BaseModel):
    """Master config object."""

    app_config: AppConfig
    model_config: NerfModelConfig

def find_config_file() -> Path:
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")

def fetch_config_from_yaml(cfg_path: Path = None) -> YAML:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")

def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # specify the data attribute from the strictyaml YAML type.
    _config = Config(
        app_config=AppConfig(**parsed_config.data),
        model_config=NerfModelConfig(**parsed_config.data),
    )

    return _config


config = create_and_validate_config()