from dataclasses import dataclass

from config.config_parsers.aggregator_parsers import AggregatorConfig
from config.config_parsers.base_config_parsers import BaseConfig
from config.config_parsers.classifier_configs import ClassifierInferConfig
from config.config_parsers.detector_parsers import DetectorInferConfig
from config.config_parsers.embedder_parsers import EmbedderInferConfig, SiameseInferConfig
from config.config_parsers.segmenter_parsers import SegmenterInferConfig
from config.config_parsers.tilerizer_parsers import TilerizerConfig


@dataclass
class XPrizeIOConfig(BaseConfig):
    raster_path: str
    output_folder: str
    coco_n_workers: int

    detector_tilerizer_config: TilerizerConfig
    detector_infer_config: DetectorInferConfig
    detector_aggregator_config: AggregatorConfig
    segmenter_tilerizer_config: TilerizerConfig
    segmenter_infer_config: SegmenterInferConfig
    segmenter_aggregator_config: AggregatorConfig
    classifier_tilerizer_config: TilerizerConfig
    classifier_embedder_config: SiameseInferConfig
    classifier_infer_config: ClassifierInferConfig

    @classmethod
    def from_dict(cls, config: dict):
        detector_config = config['detector_config']
        segmenter_config = config['segmenter_config']
        classifier_config = config['classifier_config']

        detector_tilerizer_config = TilerizerConfig.from_dict(detector_config)
        detector_infer_config = DetectorInferConfig.from_dict(detector_config)
        detector_aggregator_config = AggregatorConfig.from_dict(detector_config)
        segmenter_tilerizer_config = TilerizerConfig.from_dict(segmenter_config)
        segmenter_infer_config = SegmenterInferConfig.from_dict(segmenter_config)
        segmenter_aggregator_config = AggregatorConfig.from_dict(segmenter_config)
        classifier_tilerizer_config = TilerizerConfig.from_dict(classifier_config)
        classifier_embedder_config = SiameseInferConfig.from_dict(classifier_config)
        classifier_infer_config = ClassifierInferConfig.from_dict(classifier_config)

        xprize_io_config = config['io']

        return cls(
            raster_path=xprize_io_config['raster_path'],
            output_folder=xprize_io_config['output_folder'],
            coco_n_workers=xprize_io_config['coco_n_workers'],
            detector_tilerizer_config=detector_tilerizer_config,
            detector_infer_config=detector_infer_config,
            detector_aggregator_config=detector_aggregator_config,
            segmenter_tilerizer_config=segmenter_tilerizer_config,
            segmenter_infer_config=segmenter_infer_config,
            segmenter_aggregator_config=segmenter_aggregator_config,
            classifier_tilerizer_config=classifier_tilerizer_config,
            classifier_embedder_config=classifier_embedder_config,
            classifier_infer_config=classifier_infer_config
        )

    def to_structured_dict(self):
        config = {
            'io': {
                'raster_path': self.raster_path,
                'output_folder': self.output_folder,
                'coco_n_workers': self.coco_n_workers,
            },
            'detector_config': {
                'tilerizer': self.detector_tilerizer_config.to_structured_dict()['tilerizer'],
                'detector': self.detector_infer_config.to_structured_dict()['detector'],
                'aggregator': self.detector_aggregator_config.to_structured_dict()['aggregator']
            },
            'segmenter_config': {
                'tilerizer': self.segmenter_tilerizer_config.to_structured_dict()['tilerizer'],
                'segmenter': self.segmenter_infer_config.to_structured_dict()['segmenter'],
                'aggregator': self.segmenter_aggregator_config.to_structured_dict()['aggregator']
            },
            'classifier_config': {
                'tilerizer': self.classifier_tilerizer_config.to_structured_dict()['tilerizer'],
                'embedder': self.classifier_embedder_config.to_structured_dict()['embedder'],
                'classifier': self.classifier_infer_config.to_structured_dict()['classifier']
            }
        }

        return config

