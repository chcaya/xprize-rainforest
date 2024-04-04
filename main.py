import argparse

from config.config_parsers.aggregator_parsers import AggregatorIOConfig
from config.config_parsers.coco_to_geojson_parsers import CocoToGeojsonIOConfig
from config.config_parsers.detector_parsers import DetectorScoreIOConfig, DetectorInferIOConfig, \
    DetectorTrainIOConfig
from config.config_parsers.segmenter_parsers import SegmenterInferIOConfig
from config.config_parsers.tilerizer_parsers import TilerizerIOConfig
from config.config_parsers.xprize_parsers import XPrizeIOConfig
from mains import *
from mains.aggregator_main import aggregator_main
from mains.coco_to_geojson_main import coco_to_geojson_main
from mains.detector_mains import detector_infer_main
from mains.segmenter_main import segmenter_infer_main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, help="The task to perform.")
    parser.add_argument("--subtask", type=str, help="The subtask of the task to perform.")
    parser.add_argument("--config_path", type=str, help="Path to the appropriate .yaml config file.")
    args = parser.parse_args()

    task = args.task
    subtask = args.subtask
    config_path = args.config_path

    if task == "xprize":
        config = XPrizeIOConfig.from_config_path(config_path)
        xprize_main(config)
    if task == "tilerizer":
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
        aggregator_main(config)
    elif task == "segmenter" and subtask == "infer":
        config = SegmenterInferIOConfig.from_config_path(config_path)
        segmenter_infer_main(config)
    elif task == "coco_to_geojson":
        config = CocoToGeojsonIOConfig.from_config_path(config_path)
        coco_to_geojson_main(config)

