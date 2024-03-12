from abc import abstractmethod
from pathlib import Path

import yaml


class BaseConfig:
    @staticmethod
    def load_yaml_config(config_path):
        return yaml.safe_load(open(config_path, 'rb'))

    @abstractmethod
    def save_yaml_config(self, output_path: Path):
        pass
