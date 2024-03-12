import argparse

from config.config_parser.config_parsers import PreprocessorConfig, DetectorTrainConfig, DetectorScoreConfig, \
    XPrizeConfig, DetectorInferConfig
from mains import *
from mains.detector_mains import detector_infer_main


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
        config = XPrizeConfig.from_config_path(config_path)
        xprize_main(config)
    if task == "preprocess":
        config = PreprocessorConfig.from_config_path(config_path)
        preprocessor_main(config)
    elif task == "detector" and subtask == "train":
        config = DetectorTrainConfig.from_config_path(config_path)
        detector_train_main(config)
    elif task == "detector" and subtask == "score":
        config = DetectorScoreConfig.from_config_path(config_path)
        detector_score_main(config)
    elif task == "detector" and subtask == "infer":
        config = DetectorInferConfig.from_config_path(config_path)
        detector_infer_main(config)     # TODO where do I store the results?
