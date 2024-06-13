import argparse
import multiprocessing

from config.config_parsers.aggregator_parsers import AggregatorIOConfig
from config.config_parsers.coco_to_geopackage_parsers import CocoToGeopackageIOConfig
from config.config_parsers.detector_parsers import DetectorScoreIOConfig, DetectorInferIOConfig, \
    DetectorTrainIOConfig
from config.config_parsers.embedder_parsers import SiameseInferIOConfig
from config.config_parsers.segmenter_parsers import SegmenterInferIOConfig
from config.config_parsers.tilerizer_parsers import TilerizerIOConfig
from config.config_parsers.pipeline_parsers import PipelineXPrizeIOConfig, PipelineSegmenterIOConfig, \
    PipelineDetectorIOConfig, PipelineClassifierIOConfig

from mains import *
from mains.aggregator_mains import aggregator_main_with_coco_input
from mains.coco_to_geopackage_mains import coco_to_geopackage_main
from mains.detector_mains import detector_infer_main
from mains.embedder_mains import embedder_infer_main
from mains.segmenter_mains import segmenter_infer_main, segmenter_score_main
from mains.tilerizer_mains import tilerizer_main

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, help="The task to perform.")
    parser.add_argument("--subtask", type=str, help="The subtask of the task to perform.")
    parser.add_argument("--config_path", type=str, help="Path to the appropriate .yaml config file.")
    args = parser.parse_args()

    task = args.task
    subtask = args.subtask
    config_path = args.config_path

    if task == "pipeline" and subtask == "xprize":
        config = PipelineXPrizeIOConfig.from_config_path(config_path)
        pipeline_xprize_main(config)
    elif task == "pipeline" and subtask == "segmenter":
        config = PipelineSegmenterIOConfig.from_config_path(config_path)
        pipeline_segmenter_main(config)
    elif task == "pipeline" and subtask == "detector":
        config = PipelineDetectorIOConfig.from_config_path(config_path)
        pipeline_detector_main(config)
    elif task == "pipeline" and subtask == "classifier":
        config = PipelineClassifierIOConfig.from_config_path(config_path)
        pipeline_classifier_main(config)
    elif task == "tilerizer":
        config = TilerizerIOConfig.from_config_path(config_path)
        tilerizer_main(config)
    elif task == "detector" and subtask == "train":
        config = DetectorTrainIOConfig.from_config_path(config_path)
        detector_train_main(config)
    elif task == "detector" and subtask == "score":
        config = DetectorScoreIOConfig.from_config_path(config_path)
        detector_score_main(config)
    elif task == "detector" and subtask == "infer":
        config = DetectorInferIOConfig.from_config_path(config_path)
        detector_infer_main(config)
    elif task == "aggregator":
        config = AggregatorIOConfig.from_config_path(config_path)
        aggregator_main_with_coco_input(config)
    elif task == "segmenter" and subtask == "infer":
        config = SegmenterInferIOConfig.from_config_path(config_path)
        segmenter_infer_main(config)
    elif task == "segmenter" and subtask == "score":
        config = SegmenterScoreIOConfig.from_config_path(config_path)
        segmenter_score_main(config)
    elif task == "coco_to_geopackage":
        config = CocoToGeopackageIOConfig.from_config_path(config_path)
        coco_to_geopackage_main(config)
    elif task == "embedder" and subtask == "infer":
        config = SiameseInferIOConfig.from_config_path(config_path)
        embedder_infer_main(config)

