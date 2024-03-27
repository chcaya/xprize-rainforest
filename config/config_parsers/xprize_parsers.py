from dataclasses import dataclass

from config.config_parsers.aggregator_parsers import AggregatorConfig
from config.config_parsers.base_config_parsers import BaseConfig
from config.config_parsers.detector_parsers import DetectorInferConfig
from config.config_parsers.segmenter_parsers import SegmenterInferConfig
from config.config_parsers.tilerizer_parsers import TilerizerConfig


@dataclass
class XPrizeConfig(BaseConfig):
    raster_path: str
    output_folder: str
    coco_n_workers: int

    tilerizer_config: TilerizerConfig
    detector_infer_config: DetectorInferConfig
    aggregator_config: AggregatorConfig
    sam_infer_config: SegmenterInferConfig

    @classmethod
    def from_dict(cls, config: dict):
        tilerizer_config = TilerizerConfig.from_dict(config)
        detector_infer_config = DetectorInferConfig.from_dict(config)
        aggregator_config = AggregatorConfig.from_dict(config)
        sam_infer_config = SegmenterInferConfig.from_dict(config)

        return cls(
            raster_path=config['raster_path'],
            output_folder=config['output_folder'],
            coco_n_workers=config['coco_n_workers'],
            tilerizer_config=tilerizer_config,
            detector_infer_config=detector_infer_config,
            aggregator_config=aggregator_config,
            sam_infer_config=sam_infer_config
        )

    def to_structured_dict(self):
        config = {
            'raster_path': self.raster_path,
            'output_folder': self.output_folder,
            'coco_n_workers': self.coco_n_workers,
            'tilerizer': self.tilerizer_config.to_structured_dict(),
            'detector': self.detector_infer_config.to_structured_dict(),
            'aggregator': self.aggregator_config.to_structured_dict(),
            'sam': self.sam_infer_config.to_structured_dict()
        }

        return config

