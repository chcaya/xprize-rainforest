from dataclasses import dataclass

from config.config_parsers.aggregator_parsers import AggregatorConfig
from config.config_parsers.base_config_parsers import BaseConfig
from config.config_parsers.classifier_configs import ClassifierInferConfig
from config.config_parsers.detector_parsers import DetectorInferConfig
from config.config_parsers.embedder_parsers import SiameseInferConfig
from config.config_parsers.segmenter_parsers import SegmenterInferConfig
from config.config_parsers.tilerizer_parsers import TilerizerConfig


@dataclass
class PipelineDetectorConfig(BaseConfig):
    save_detector_intermediate_output: bool
    detector_tilerizer_config: TilerizerConfig
    detector_infer_config: DetectorInferConfig
    detector_aggregator_config: AggregatorConfig

    @classmethod
    def from_dict(cls, config: dict):
        pipeline_detector_config = config['pipeline_detector']

        save_detector_intermediate_output = pipeline_detector_config['save_detector_intermediate_output']
        detector_tilerizer_config = TilerizerConfig.from_dict(pipeline_detector_config)
        detector_infer_config = DetectorInferConfig.from_dict(pipeline_detector_config)
        detector_aggregator_config = AggregatorConfig.from_dict(pipeline_detector_config)

        return cls(
            save_detector_intermediate_output=save_detector_intermediate_output,
            detector_tilerizer_config=detector_tilerizer_config,
            detector_infer_config=detector_infer_config,
            detector_aggregator_config=detector_aggregator_config,
        )

    def to_structured_dict(self):
        config = {
            'pipeline_detector': {
                'save_segmenter_intermediate_output': self.save_detector_intermediate_output,
                'tilerizer': self.detector_tilerizer_config.to_structured_dict()['tilerizer'],
                'detector': self.detector_infer_config.to_structured_dict()['detector'],
                'aggregator': self.detector_aggregator_config.to_structured_dict()['aggregator']
            }
        }

        return config


@dataclass
class PipelineDetectorIOConfig(PipelineDetectorConfig):
    raster_path: str
    output_folder: str
    coco_n_workers: int

    @classmethod
    def from_dict(cls, config: dict):
        parent_config = PipelineDetectorConfig.from_dict(config)

        pipeline_detector_io_config = config['io']

        return cls(
            **parent_config.as_dict(),
            raster_path=pipeline_detector_io_config['raster_path'],
            output_folder=pipeline_detector_io_config['output_folder'],
            coco_n_workers=pipeline_detector_io_config['coco_n_workers'],
        )

    def to_structured_dict(self):
        config = super().to_structured_dict()
        config['pipeline_detector']['io'] = {
            'raster_path': self.raster_path,
            'output_folder': self.output_folder,
            'coco_n_workers': self.coco_n_workers,
        }

        return config


@dataclass
class PipelineSegmenterConfig(BaseConfig):
    save_segmenter_intermediate_output: bool
    segmenter_tilerizer_config: TilerizerConfig
    segmenter_infer_config: SegmenterInferConfig
    segmenter_aggregator_config: AggregatorConfig

    @classmethod
    def from_dict(cls, config: dict):
        pipeline_segmenter_config = config['pipeline_segmenter']

        save_segmenter_intermediate_output = pipeline_segmenter_config['save_segmenter_intermediate_output']
        segmenter_tilerizer_config = TilerizerConfig.from_dict(pipeline_segmenter_config)
        segmenter_infer_config = SegmenterInferConfig.from_dict(pipeline_segmenter_config)
        segmenter_aggregator_config = AggregatorConfig.from_dict(pipeline_segmenter_config)

        return cls(
            save_segmenter_intermediate_output=save_segmenter_intermediate_output,
            segmenter_tilerizer_config=segmenter_tilerizer_config,
            segmenter_infer_config=segmenter_infer_config,
            segmenter_aggregator_config=segmenter_aggregator_config,
        )

    def to_structured_dict(self):
        config = {
            'pipeline_segmenter': {
                'save_segmenter_intermediate_output': self.save_segmenter_intermediate_output,
                'tilerizer': self.segmenter_tilerizer_config.to_structured_dict()['tilerizer'],
                'segmenter': self.segmenter_infer_config.to_structured_dict()['segmenter'],
                'aggregator': self.segmenter_aggregator_config.to_structured_dict()['aggregator']
            }
        }

        return config


@dataclass
class PipelineSegmenterIOConfig(PipelineSegmenterConfig):
    raster_path: str
    boxes_geopackage_path: str
    output_folder: str

    @classmethod
    def from_dict(cls, config: dict):
        parent_config = PipelineSegmenterConfig.from_dict(config)

        pipeline_segmenter_io_config = config['io']

        return cls(
            **parent_config.as_dict(),
            raster_path=pipeline_segmenter_io_config['raster_path'],
            boxes_geopackage_path=pipeline_segmenter_io_config['boxes_geopackage_path'],
            output_folder=pipeline_segmenter_io_config['output_folder'],
        )

    def to_structured_dict(self):
        config = super().to_structured_dict()
        config['pipeline_segmenter']['io'] = {
            'raster_path': self.raster_path,
            'boxes_geopackage_path': self.boxes_geopackage_path,
            'output_folder': self.output_folder,
        }

        return config


@dataclass
class PipelineClassifierConfig(BaseConfig):
    classifier_tilerizer_config: TilerizerConfig
    classifier_embedder_config: SiameseInferConfig
    classifier_infer_config: ClassifierInferConfig

    @classmethod
    def from_dict(cls, config: dict):
        pipeline_classifier_config = config['pipeline_classifier']

        classifier_tilerizer_config = TilerizerConfig.from_dict(pipeline_classifier_config)
        classifier_embedder_config = SiameseInferConfig.from_dict(pipeline_classifier_config)
        classifier_infer_config = ClassifierInferConfig.from_dict(pipeline_classifier_config)

        return cls(
            classifier_tilerizer_config=classifier_tilerizer_config,
            classifier_embedder_config=classifier_embedder_config,
            classifier_infer_config=classifier_infer_config,
        )

    def to_structured_dict(self):
        config = {
            'pipeline_classifier': {
                'tilerizer': self.classifier_tilerizer_config.to_structured_dict()['tilerizer'],
                'embedder': self.classifier_embedder_config.to_structured_dict()['embedder'],
                'classifier': self.classifier_infer_config.to_structured_dict()['classifier']
            }
        }

        return config


@dataclass
class PipelineClassifierIOConfig(PipelineClassifierConfig):
    raster_path: str
    segmentations_geopackage_path: str
    output_folder: str

    @classmethod
    def from_dict(cls, config: dict):
        parent_config = PipelineClassifierConfig.from_dict(config)

        pipeline_classifier_io_config = config['io']

        return cls(
            **parent_config.as_dict(),
            raster_path=pipeline_classifier_io_config['raster_path'],
            segmentations_geopackage_path=pipeline_classifier_io_config['segmentations_geopackage_path'],
            output_folder=pipeline_classifier_io_config['output_folder'],
        )

    def to_structured_dict(self):
        config = super().to_structured_dict()
        config['pipeline_classifier']['io'] = {
            'raster_path': self.raster_path,
            'segmentations_geopackage_path': self.segmentations_geopackage_path,
            'output_folder': self.output_folder,
        }

        return config


@dataclass
class PipelineXPrizeIOConfig(BaseConfig):
    raster_path: str
    output_folder: str
    coco_n_workers: int

    pipeline_detector_config: PipelineDetectorConfig
    pipeline_segmenter_config: PipelineSegmenterConfig
    pipeline_classifier_config: PipelineClassifierConfig

    @classmethod
    def from_dict(cls, config: dict):
        pipeline_detector_config = PipelineDetectorConfig.from_dict(config)
        pipeline_segmenter_config = PipelineSegmenterConfig.from_dict(config)
        pipeline_classifier_config = PipelineClassifierConfig.from_dict(config)

        pipeline_xprize_io_config = config['io']

        return cls(
            raster_path=pipeline_xprize_io_config['raster_path'],
            output_folder=pipeline_xprize_io_config['output_folder'],
            coco_n_workers=pipeline_xprize_io_config['coco_n_workers'],
            pipeline_detector_config=pipeline_detector_config,
            pipeline_segmenter_config=pipeline_segmenter_config,
            pipeline_classifier_config=pipeline_classifier_config,
        )

    def to_structured_dict(self):
        config = {
            'io': {
                'raster_path': self.raster_path,
                'output_folder': self.output_folder,
                'coco_n_workers': self.coco_n_workers,
            },
            'pipeline_detector': self.pipeline_detector_config.to_structured_dict()['pipeline_detector'],
            'pipeline_segmenter': self.pipeline_segmenter_config.to_structured_dict()['pipeline_segmenter'],
            'classifier_config': self.pipeline_classifier_config.to_structured_dict()['pipeline_classifier']
        }

        return config
