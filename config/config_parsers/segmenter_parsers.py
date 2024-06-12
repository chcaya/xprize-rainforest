from dataclasses import dataclass

from config.config_parsers.base_config_parsers import BaseConfig
from config.config_parsers.tilerizer_parsers import RasterResolutionConfig


@dataclass
class SegmenterInferConfig(BaseConfig):
    model_type: str
    checkpoint_path: str
    simplify_tolerance: float
    box_padding_percentage: float
    n_postprocess_workers: int
    box_batch_size: int

    @classmethod
    def from_dict(cls, config: dict):
        segmenter_config = config['segmenter']['infer']

        return cls(
            model_type=segmenter_config['model_type'],
            checkpoint_path=segmenter_config['checkpoint_path'],
            simplify_tolerance=segmenter_config['simplify_tolerance'],
            box_padding_percentage=segmenter_config['box_padding_percentage'],
            n_postprocess_workers=segmenter_config['n_postprocess_workers'],
            box_batch_size=segmenter_config['box_batch_size']
        )

    def to_structured_dict(self):
        config = {
            'segmenter': {
                'infer': {
                    'model_type': self.model_type,
                    'checkpoint_path': self.checkpoint_path,
                    'simplify_tolerance': self.simplify_tolerance,
                    'box_padding_percentage': self.box_padding_percentage,
                    'n_postprocess_workers': self.n_postprocess_workers,
                    'box_batch_size': self.box_batch_size
                }
            }
        }

        return config


@dataclass
class SegmenterInferIOConfig(SegmenterInferConfig):
    coco_path: str
    input_tiles_root: str
    output_folder: str

    @classmethod
    def from_dict(cls, config: dict):
        parent_config = SegmenterInferConfig.from_dict(config)
        segmenter_io_config = config['segmenter']['infer']['io']
        return cls(
            **parent_config.as_dict(),
            coco_path=segmenter_io_config['coco_path'],
            input_tiles_root=segmenter_io_config['input_tiles_root'],
            output_folder=segmenter_io_config['output_folder'],
        )

    def to_structured_dict(self):
        config = super().to_structured_dict()
        config['segmenter']['infer']['io'] = {
            'coco_path': self.coco_path,
            'input_tiles_root': self.input_tiles_root,
            'output_folder': self.output_folder
        }


@dataclass
class SegmenterScoreIOConfig(BaseConfig):
    truth_geopackage_path: str
    predictions_geopackage_path: str
    class_column_name: str

    @classmethod
    def from_dict(cls, config: dict):
        segmenter_io_config = config['segmenter']['score']['io']
        return cls(
            truth_geopackage_path=segmenter_io_config['truth_geopackage_path'],
            predictions_geopackage_path=segmenter_io_config['predictions_geopackage_path'],
            class_column_name=segmenter_io_config['class_column_name']
        )

    def to_structured_dict(self):
        config = {
            'segmenter': {
                'score': {
                    'io': {
                        'truth_geopackage_path': self.truth_geopackage_path,
                        'predictions_geopackage_path': self.predictions_geopackage_path,
                        'class_column_name': self.class_column_name
                    }
                }
            }
        }

        return config
