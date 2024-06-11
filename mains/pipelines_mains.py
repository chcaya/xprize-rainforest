from pathlib import Path

from config.config_parsers.pipeline_parsers import PipelineXPrizeIOConfig, PipelineSegmenterIOConfig, \
    PipelineDetectorIOConfig, PipelineClassifierIOConfig
from engine.pipelines.pipeline_classifier import PipelineClassifier
from engine.pipelines.pipeline_detector import PipelineDetector
from engine.pipelines.pipeline_infer import PipelineXPrize
from engine.pipelines.pipeline_segmenter import PipelineSegmenter


def pipeline_xprize_main(config: PipelineXPrizeIOConfig):
    output_folder = Path(config.output_folder)
    output_folder.mkdir(exist_ok=False, parents=True)

    pipeline = PipelineXPrize.from_config(config)
    pipeline.run()

    config.save_yaml_config(output_folder / 'pipeline_xprize_config.yaml')


def pipeline_detector_main(config: PipelineDetectorIOConfig):
    output_folder = Path(config.output_folder)
    output_folder.mkdir(exist_ok=False, parents=True)

    pipeline = PipelineDetector.from_config(config)
    pipeline.run()

    config.save_yaml_config(output_folder / 'pipeline_detector_config.yaml')


def pipeline_segmenter_main(config: PipelineSegmenterIOConfig):
    output_folder = Path(config.output_folder)
    output_folder.mkdir(exist_ok=False, parents=True)

    pipeline = PipelineSegmenter.from_config(config)
    pipeline.run()

    config.save_yaml_config(output_folder / 'pipeline_segmenter_config.yaml')


def pipeline_classifier_main(config: PipelineClassifierIOConfig):
    output_folder = Path(config.output_folder)
    output_folder.mkdir(exist_ok=False, parents=True)

    pipeline = PipelineClassifier.from_config(config)
    pipeline.run()

    config.save_yaml_config(output_folder / 'pipeline_classifier_config.yaml')
