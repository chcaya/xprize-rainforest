from config.config_parser.config_parsers import DetectorTrainingConfig


def detector_training_main(config_path: str):
    config = DetectorTrainingConfig(config_path)
