from pathlib import Path

from geodataset.dataset import DetectionLabeledRasterCocoDataset, UnlabeledRasterDataset

from engine.detector.utils import collate_fn_detection, collate_fn_images
from config.config_parser.config_parsers import DetectorTrainConfig, DetectorScoreConfig, DetectorInferConfig
from engine.detector.detector_pipelines import DetectorTrainPipeline, DetectorScorePipeline, DetectorInferencePipeline


def detector_train_main(config: DetectorTrainConfig):
    trainer = DetectorTrainPipeline.from_config(config)
    train_ds = DetectionLabeledRasterCocoDataset(root_path=Path(config.config.data_root_path),
                                                 fold="train",
                                                 transform=DetectorTrainPipeline.get_data_augmentation_transform())
    valid_ds = DetectionLabeledRasterCocoDataset(root_path=Path(config.config.data_root_path),
                                                 fold="valid",
                                                 transform=None)
    trainer.train(train_ds=train_ds, valid_ds=valid_ds, collate_fn=collate_fn_detection)


def detector_score_main(config: DetectorScoreConfig):
    scorer = DetectorScorePipeline.from_config(config)
    test_ds = DetectionLabeledRasterCocoDataset(root_path=Path(config.data_root_path),
                                                fold="test",
                                                transform=None)  # No augmentation for testing
    scorer.score(test_ds=test_ds, collate_fn=collate_fn_detection)


def detector_infer_main(config: DetectorInferConfig):
    inferer = DetectorInferencePipeline.from_config(config)
    infer_ds = UnlabeledRasterDataset(root_path=Path(config.data_root_path),
                                      fold="infer",
                                      transform=None)  # No augmentation for testing
    inferer.infer(infer_ds=infer_ds, collate_fn=collate_fn_images)

