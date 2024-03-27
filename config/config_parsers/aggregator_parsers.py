from dataclasses import dataclass

from config.config_parsers.base_config_parsers import BaseConfig


@dataclass
class AggregatorConfig(BaseConfig):
    score_threshold: float
    nms_threshold: float
    nms_algorithm: str

    @classmethod
    def from_dict(cls, config: dict):
        aggregator_config = config['aggregator']
        return cls(
            score_threshold=aggregator_config['score_threshold'],
            nms_threshold=aggregator_config['nms_threshold'],
            nms_algorithm=aggregator_config['nms_algorithm']
        )

    def to_structured_dict(self):
        config = {
            'aggregator': {
                'score_threshold': self.score_threshold,
                'nms_threshold': self.nms_threshold,
                'nms_algorithm': self.nms_algorithm
            }
        }

        return config


@dataclass
class AggregatorCLIConfig(AggregatorConfig):
    input_tiles_root: str
    coco_path: str
    output_folder: str

    @classmethod
    def from_dict(cls, config: dict):
        parent_config = AggregatorConfig.from_dict(config)
        aggregator_io_config = config['aggregator']['io']
        return cls(
            **parent_config.as_dict(),
            input_tiles_root=aggregator_io_config['input_tiles_root'],
            coco_path=aggregator_io_config['coco_path'],
            output_folder=aggregator_io_config['output_folder'],
        )

    def to_structured_dict(self):
        config = super().to_structured_dict()
        config['aggregator']['io'] = {
            'input_tiles_root': self.input_tiles_root,
            'coco_path': self.coco_path,
            'output_folder': self.output_folder
        }
