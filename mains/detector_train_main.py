from config.config_parser.config_parsers import DetectorTrainingConfig
from engine.detector.train_pipeline import DetectorTrainingPipeline


def detector_training_main(config_path: str):
    config = DetectorTrainingConfig(config_path)

    trainer = DetectorTrainingPipeline(config=config)
    trainer.train()
