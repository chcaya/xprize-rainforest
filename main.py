from engine.detector.train_pipeline import DetectorTrainingPipeline
from config.config_loader import DEFAULT_CONFIG

if __name__ == "__main__":
    training_pipeline = DetectorTrainingPipeline(config=DEFAULT_CONFIG)
    training_pipeline.plot_data_to_disk(show_images=False)
