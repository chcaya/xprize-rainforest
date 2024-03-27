from dataclasses import dataclass

from config.config_parsers.base_config_parsers import BaseConfig


@dataclass
class SegmenterInferConfig(BaseConfig):
    model_type: str
    checkpoint_path: str
    simplify_tolerance: float
    padding_percentage: float
    min_pixel_padding: int

    @classmethod
    def from_dict(cls, config: dict):
        segmenter_config = config['segmenter']

        return cls(
            model_type=segmenter_config['model_type'],
            checkpoint_path=segmenter_config['checkpoint_path'],
            simplify_tolerance=segmenter_config['simplify_tolerance'],
            padding_percentage=segmenter_config['padding_percentage'],
            min_pixel_padding=segmenter_config['min_pixel_padding']
        )

    def to_structured_dict(self):
        config = {
            'segmenter': {
                'model_type': self.model_type,
                'checkpoint_path': self.checkpoint_path,
                'simplify_tolerance': self.simplify_tolerance,
                'padding_percentage': self.padding_percentage,
                'min_pixel_padding': self.min_pixel_padding
            }
        }

        return config


@dataclass
class SegmenterInferCLIConfig(SegmenterInferConfig):
    raster_path: str
    boxes_path: str
    output_folder: str

    @classmethod
    def from_dict(cls, config: dict):
        parent_config = SegmenterInferConfig.from_dict(config)
        segmenter_io_config = config['segmenter']['io']
        return cls(
            **parent_config.as_dict(),
            raster_path=segmenter_io_config['raster_path'],
            boxes_path=segmenter_io_config['boxes_path'],
            output_folder=segmenter_io_config['output_folder'],
        )

    def to_structured_dict(self):
        config = super().to_structured_dict()
        config['segmenter']['io'] = {
            'raster_path': self.raster_path,
            'boxes_path': self.boxes_path,
            'output_folder': self.output_folder
        }
