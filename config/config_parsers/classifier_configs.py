from dataclasses import dataclass

from config.config_parsers.base_config_parsers import BaseConfig, BaseIntermediateConfig


@dataclass
class ClassifierArchitectureConfig(BaseIntermediateConfig):
    architecture_name: str

    @classmethod
    def from_dict(cls, config: dict):
        return cls(**config)

    def to_structured_dict(self):
        config = {
            'architecture_name': self.architecture_name,
        }

        return config


@dataclass
class ClassifierInferConfig(BaseConfig):
    scaler_checkpoint_path: str
    classifier_checkpoint_path: str
    architecture_config: ClassifierArchitectureConfig

    @classmethod
    def from_dict(cls, config: dict):
        classifier_infer_config = config['classifier']['infer']

        return cls(
            scaler_checkpoint_path=classifier_infer_config['scaler_checkpoint_path'],
            classifier_checkpoint_path=classifier_infer_config['classifier_checkpoint_path'],
            architecture_config=ClassifierArchitectureConfig.from_dict(classifier_infer_config['architecture']),
        )

    def to_structured_dict(self) -> dict:
        config = {
            'classifier': {
                'infer': {
                    'scaler_checkpoint_path': self.scaler_checkpoint_path,
                    'classifier_checkpoint_path': self.classifier_checkpoint_path,
                    'architecture': self.architecture_config.to_structured_dict(),
                }
            }
        }

        return config
