from dataclasses import dataclass

from config.base.base_config import BaseConfig


@dataclass
class DetectorTrainingConfig(BaseConfig):
    output_folder: str
    output_name: str

    train_datasets: dict
    valid_datasets: dict
    test_datasets: dict

    chip_size: int
    chip_stride: int

    batch_size: int
    learning_rate: float
    n_epochs: int
    rcnn_backbone_model_resnet_name: str
    rcnn_backbone_model_pretrained: bool
    backbone_resnet_out_channels: int
    box_predictions_per_image: int

    def __init__(self, config_path: str):
        super().__init__()
        config = self.load_yaml_config(config_path)

        self.output_folder = config['output']['output_folder']
        self.output_name = config['output']['output_name']

        self.train_datasets = config['datasets']['train']
        self.valid_datasets = config['datasets']['valid']
        self.test_datasets = config['datasets']['test']

        self.chip_size = config['tiles']['chip_size']
        self.chip_stride = config['tiles']['chip_stride']

        self.batch_size = config['model_config']['batch_size']
        self.learning_rate = config['model_config']['learning_rate']
        self.n_epochs = config['model_config']['n_epochs']
        self.rcnn_backbone_model_resnet_name = config['model_config']['rcnn_backbone_model_resnet_name']
        self.rcnn_backbone_model_pretrained = config['model_config']['rcnn_backbone_model_pretrained']
        self.backbone_resnet_out_channels = config['model_config']['backbone_resnet_out_channels']
        self.box_predictions_per_image = config['model_config']['box_predictions_per_image']


def detector_training_main(config_path: str):
    config = DetectorTrainingConfig(config_path)
