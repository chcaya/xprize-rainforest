from dataclasses import dataclass
from typing import List, Tuple

from config.config_parsers.aggregator_parsers import AggregatorConfig
from config.config_parsers.base_config_parsers import BaseConfig
from config.config_parsers.classifier_configs import ClassifierInferConfig
from config.config_parsers.detector_parsers import DetectorInferConfig
from config.config_parsers.embedder_parsers import SiameseInferConfig, ContrastiveInferConfig, DINOv2InferConfig
from config.config_parsers.segmenter_parsers import SegmenterInferConfig
from config.config_parsers.tilerizer_parsers import TilerizerConfig, TilerizerNoAoiConfig


@dataclass
class PipelineDetectorConfig(BaseConfig):
    save_detector_intermediate_output: bool
    detector_tilerizer_config: TilerizerNoAoiConfig
    detector_infer_config: DetectorInferConfig
    detector_aggregator_config: AggregatorConfig

    @classmethod
    def from_dict(cls, config: dict):
        pipeline_detector_config = config['pipeline_detector']

        save_detector_intermediate_output = pipeline_detector_config['save_detector_intermediate_output']
        detector_tilerizer_config = TilerizerNoAoiConfig.from_dict(pipeline_detector_config)
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
    aoi_geopackage_path: str
    output_folder: str
    coco_n_workers: int

    @classmethod
    def from_dict(cls, config: dict):
        parent_config = PipelineDetectorConfig.from_dict(config)

        pipeline_detector_io_config = config['io']

        return cls(
            **parent_config.as_dict(),
            raster_path=pipeline_detector_io_config['raster_path'],
            aoi_geopackage_path=pipeline_detector_io_config['aoi_geopackage_path'],
            output_folder=pipeline_detector_io_config['output_folder'],
            coco_n_workers=pipeline_detector_io_config['coco_n_workers'],
        )

    def to_structured_dict(self):
        config = super().to_structured_dict()
        config['pipeline_detector']['io'] = {
            'raster_path': self.raster_path,
            'aoi_geopackage_path': self.aoi_geopackage_path,
            'output_folder': self.output_folder,
            'coco_n_workers': self.coco_n_workers,
        }

        return config


@dataclass
class PipelineSegmenterConfig(BaseConfig):
    save_segmenter_intermediate_output: bool
    segmenter_tilerizer_config: TilerizerNoAoiConfig
    segmenter_infer_config: SegmenterInferConfig
    segmenter_aggregator_config: AggregatorConfig

    @classmethod
    def from_dict(cls, config: dict):
        pipeline_segmenter_config = config['pipeline_segmenter']

        save_segmenter_intermediate_output = pipeline_segmenter_config['save_segmenter_intermediate_output']
        segmenter_tilerizer_config = TilerizerNoAoiConfig.from_dict(pipeline_segmenter_config)
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
    aoi_geopackage_path: str
    output_folder: str

    @classmethod
    def from_dict(cls, config: dict):
        parent_config = PipelineSegmenterConfig.from_dict(config)

        pipeline_segmenter_io_config = config['io']

        return cls(
            **parent_config.as_dict(),
            raster_path=pipeline_segmenter_io_config['raster_path'],
            boxes_geopackage_path=pipeline_segmenter_io_config['boxes_geopackage_path'],
            aoi_geopackage_path=pipeline_segmenter_io_config['aoi_geopackage_path'],
            output_folder=pipeline_segmenter_io_config['output_folder'],
        )

    def to_structured_dict(self):
        config = super().to_structured_dict()
        config['pipeline_segmenter']['io'] = {
            'raster_path': self.raster_path,
            'boxes_geopackage_path': self.boxes_geopackage_path,
            'aoi_geopackage_path': self.aoi_geopackage_path,
            'output_folder': self.output_folder,
        }

        return config


@dataclass
class PipelineClassifierConfig(BaseConfig):
    classifier_tilerizer_config: TilerizerNoAoiConfig
    classifier_contrastive_embedder_config: ContrastiveInferConfig or None
    classifier_dinov2_embedder_config: DINOv2InferConfig or None

    @classmethod
    def from_dict(cls, config: dict):
        pipeline_classifier_config = config['pipeline_classifier']

        classifier_tilerizer_config = TilerizerNoAoiConfig.from_dict(pipeline_classifier_config)
        embedder_infer_config = pipeline_classifier_config['embedder']['infer']
        if 'contrastive' in embedder_infer_config:
            classifier_contrastive_embedder_config = ContrastiveInferConfig.from_dict(pipeline_classifier_config)
        else:
            classifier_contrastive_embedder_config = None

        if 'dinov2' in embedder_infer_config:
            classifier_dinov2_embedder_config = DINOv2InferConfig.from_dict(pipeline_classifier_config)
        else:
            classifier_dinov2_embedder_config = None

        return cls(
            classifier_tilerizer_config=classifier_tilerizer_config,
            classifier_contrastive_embedder_config=classifier_contrastive_embedder_config,
            classifier_dinov2_embedder_config=classifier_dinov2_embedder_config,
        )

    def to_structured_dict(self):
        config = {
            'pipeline_classifier': {
                'tilerizer': self.classifier_tilerizer_config.to_structured_dict()['tilerizer'],
                'embedder': {
                    'infer': {}
                }
            }
        }

        if self.classifier_contrastive_embedder_config is not None:
            config['pipeline_classifier']['embedder']['infer']['contrastive'] = self.classifier_contrastive_embedder_config.to_structured_dict()['embedder']['infer']['contrastive']
        if self.classifier_dinov2_embedder_config is not None:
            config['pipeline_classifier']['embedder']['infer']['dinov2'] = self.classifier_dinov2_embedder_config.to_structured_dict()['embedder']['infer']['dinov2']

        return config


@dataclass
class PipelineClassifierIOConfig(PipelineClassifierConfig):
    raster_path: str
    aoi_geopackage_path: str
    segmentations_geopackage_path: str
    output_folder: str
    day_month_year: Tuple[int, int, int] or None

    @classmethod
    def from_dict(cls, config: dict):
        parent_config = PipelineClassifierConfig.from_dict(config)

        pipeline_classifier_io_config = config['io']

        return cls(
            **parent_config.as_dict(),
            raster_path=pipeline_classifier_io_config['raster_path'],
            aoi_geopackage_path=pipeline_classifier_io_config['aoi_geopackage_path'],
            segmentations_geopackage_path=pipeline_classifier_io_config['segmentations_geopackage_path'],
            output_folder=pipeline_classifier_io_config['output_folder'],
            day_month_year=pipeline_classifier_io_config['day_month_year'],
        )

    def to_structured_dict(self):
        config = super().to_structured_dict()
        config['pipeline_classifier']['io'] = {
            'raster_path': self.raster_path,
            'aoi_geopackage_path': self.aoi_geopackage_path,
            'segmentations_geopackage_path': self.segmentations_geopackage_path,
            'output_folder': self.output_folder,
            'day_month_year': self.day_month_year
        }

        return config


@dataclass
class PipelineXPrizeIOConfig(BaseConfig):
    raster_path: str
    aoi_geopackage_path: str
    output_folder: str
    coco_n_workers: int
    day_month_year: Tuple[int, int, int] or None

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
            aoi_geopackage_path=pipeline_xprize_io_config['aoi_geopackage_path'],
            output_folder=pipeline_xprize_io_config['output_folder'],
            coco_n_workers=pipeline_xprize_io_config['coco_n_workers'],
            day_month_year=pipeline_xprize_io_config['day_month_year'],
            pipeline_detector_config=pipeline_detector_config,
            pipeline_segmenter_config=pipeline_segmenter_config,
            pipeline_classifier_config=pipeline_classifier_config,
        )

    def to_structured_dict(self):
        config = {
            'io': {
                'raster_path': self.raster_path,
                'aoi_geopackage_path': self.aoi_geopackage_path,
                'output_folder': self.output_folder,
                'coco_n_workers': self.coco_n_workers,
                'day_month_year': self.day_month_year
            },
            'pipeline_detector': self.pipeline_detector_config.to_structured_dict()['pipeline_detector'],
            'pipeline_segmenter': self.pipeline_segmenter_config.to_structured_dict()['pipeline_segmenter'],
            'classifier_config': self.pipeline_classifier_config.to_structured_dict()['pipeline_classifier']
        }

        return config
