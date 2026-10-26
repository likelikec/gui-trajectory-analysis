import os
from pathlib import Path
import yaml


def load_config(config_path="./config.yaml"):
    config_file = Path(config_path)
    if not config_file.is_absolute():
        repo_root = Path(__file__).resolve().parents[1]
        config_file = repo_root / config_file
    configs = dict(os.environ)
    with open(config_file, "r", encoding="utf-8") as file:
        yaml_data = yaml.safe_load(file)
    configs.update(yaml_data)
    return configs
