from abc import abstractmethod, ABC
from dataclasses import fields, dataclass
from pathlib import Path

import yaml


@dataclass
class BaseIntermediateConfig(ABC):
    @staticmethod
    def load_yaml_config(config_path) -> dict:
        return yaml.safe_load(open(config_path, 'rb'))

    @abstractmethod
    def to_structured_dict(self) -> dict:
        pass

    def as_dict(self):
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass
class BaseConfig(BaseIntermediateConfig):
    @staticmethod
    def load_yaml_config(config_path) -> dict:
        return yaml.safe_load(open(config_path, 'rb'))

    @classmethod
    def from_config_path(cls, config_path: str):
        config = cls.load_yaml_config(config_path)
        return cls.from_dict(config)

    @classmethod
    def from_dict(cls, config: dict):
        pass

    @abstractmethod
    def to_structured_dict(self) -> dict:
        pass

    def save_yaml_config(self, output_path: Path):
        with output_path.open('w') as file:
            yaml.dump(self.to_structured_dict(), file, sort_keys=False)
