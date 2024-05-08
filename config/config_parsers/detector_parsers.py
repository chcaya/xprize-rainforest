from dataclasses import dataclass

from config.config_parsers.base_config_parsers import BaseConfig, BaseIntermediateConfig


@dataclass
class DetectorBaseParamsConfig(BaseIntermediateConfig):
    batch_size: int
    box_predictions_per_image: int

    @classmethod
    def from_dict(cls, config: dict):
        return cls(**config)

    def to_structured_dict(self):
        config = {
            'batch_size': self.batch_size,
            'box_predictions_per_image': self.box_predictions_per_image,
        }

        return config


@dataclass
class DetectorArchitectureConfig(BaseIntermediateConfig):
    architecture_name: str
    rcnn_backbone_model_resnet_name: str

    @classmethod
    def from_dict(cls, config: dict):
        return cls(**config)

    def to_structured_dict(self):
        config = {
            'architecture_name': self.architecture_name,
            'rcnn_backbone_model_resnet_name': self.rcnn_backbone_model_resnet_name,
        }

        return config


@dataclass
class DetectorTrainIOConfig(BaseConfig):
    data_root: str
    output_folder: str
    output_name: str
    train_aoi_name: str
    valid_aoi_name: str

    train_log_interval: int

    base_params_config: DetectorBaseParamsConfig
    architecture_config: DetectorArchitectureConfig

    grad_accumulation_steps: int
    rcnn_backbone_model_pretrained: bool
    start_checkpoint_state_dict_path: str or None
    learning_rate: float
    n_epochs: int
    save_model_every_n_epoch: int
    backbone_resnet_out_channels: int
    scheduler_step_size: int
    scheduler_gamma: float

    @classmethod
    def from_dict(cls, config: dict):
        detector_train_config = config['detector']['train']
        detector_train_io = detector_train_config['io']
        detector_train_model_config = detector_train_config['model_config']

        base_params_config = DetectorBaseParamsConfig.from_dict(detector_train_config['base_params'])
        architecture_config = DetectorArchitectureConfig.from_dict(detector_train_config['architecture'])

        return cls(
            base_params_config=base_params_config,
            architecture_config=architecture_config,
            data_root=detector_train_io['data_root'],
            train_aoi_name=detector_train_io['train_aoi_name'],
            valid_aoi_name=detector_train_io['valid_aoi_name'],
            output_folder=detector_train_io['output_folder'],
            output_name=detector_train_io['output_name'],
            train_log_interval=int(detector_train_io['train_log_interval']),
            grad_accumulation_steps=int(detector_train_model_config['grad_accumulation_steps']),
            rcnn_backbone_model_pretrained=bool(detector_train_model_config['rcnn_backbone_model_pretrained']),
            start_checkpoint_state_dict_path=detector_train_model_config['start_checkpoint_state_dict_path'],
            learning_rate=float(detector_train_model_config['learning_rate']),
            n_epochs=int(detector_train_model_config['n_epochs']),
            save_model_every_n_epoch=int(detector_train_model_config['save_model_every_n_epoch']),
            backbone_resnet_out_channels=int(detector_train_model_config['backbone_resnet_out_channels']),
            scheduler_step_size=int(detector_train_model_config['scheduler_step_size']),
            scheduler_gamma=float(detector_train_model_config['scheduler_gamma']),
        )

    def to_structured_dict(self):
        config = {
            'detector': {
                'train': {
                    'io': {
                        'data_root': self.data_root,
                        'train_aoi_name': self.train_aoi_name,
                        'valid_aoi_name': self.valid_aoi_name,
                        'output_folder': self.output_folder,
                        'output_name': self.output_name,
                        'train_log_interval': self.train_log_interval,
                    },
                    'base_params': self.base_params_config.to_structured_dict(),
                    'architecture': self.architecture_config.to_structured_dict(),
                    'model_config': {
                        'grad_accumulation_steps': self.grad_accumulation_steps,
                        'rcnn_backbone_model_pretrained': self.rcnn_backbone_model_pretrained,
                        'start_checkpoint_state_dict_path': self.start_checkpoint_state_dict_path,
                        'learning_rate': self.learning_rate,
                        'n_epochs': self.n_epochs,
                        'save_model_every_n_epoch': self.save_model_every_n_epoch,
                        'backbone_resnet_out_channels': self.backbone_resnet_out_channels,
                        'scheduler_step_size': self.scheduler_step_size,
                        'scheduler_gamma': self.scheduler_gamma,
                    }
                }
            }
        }

        return config


@dataclass
class DetectorScoreConfig(BaseConfig):
    checkpoint_state_dict_path: str

    base_params_config: DetectorBaseParamsConfig
    architecture_config: DetectorArchitectureConfig

    @classmethod
    def from_dict(cls, config: dict):
        detector_score_config = config['detector']['score']
        detector_score_io_config = detector_score_config['io']

        base_params_config = DetectorBaseParamsConfig.from_dict(detector_score_config['base_params'])
        architecture_config = DetectorArchitectureConfig.from_dict(detector_score_config['architecture'])
        return cls(
            checkpoint_state_dict_path=detector_score_io_config['checkpoint_state_dict_path'],

            base_params_config=base_params_config,
            architecture_config=architecture_config,
        )

    def to_structured_dict(self):
        config = {
            'detector': {
                'score': {
                    'io': {
                        'checkpoint_state_dict_path': self.checkpoint_state_dict_path,
                    },
                    'base_params': self.base_params_config.to_structured_dict(),
                    'architecture': self.architecture_config.to_structured_dict(),
                }
            }
        }

        return config


@dataclass
class DetectorScoreIOConfig(DetectorScoreConfig):
    data_root: str
    score_aoi_name: str
    output_folder: str
    coco_n_workers: int

    @classmethod
    def from_dict(cls, config: dict):
        parent_config = DetectorScoreConfig.from_dict(config)
        detector_score_io_config = config['detector']['score']['io']

        return cls(
            **parent_config.as_dict(),
            data_root=detector_score_io_config['data_root'],
            score_aoi_name=detector_score_io_config['score_aoi_name'],
            output_folder=detector_score_io_config['output_folder'],
            coco_n_workers=detector_score_io_config['coco_n_workers']
        )

    def to_structured_dict(self):
        config = super().to_structured_dict()
        config['detector']['score']['io']['data_root'] = self.data_root
        config['detector']['score']['io']['score_aoi_name'] = self.score_aoi_name
        config['detector']['score']['io']['output_folder'] = self.output_folder
        config['detector']['score']['io']['coco_n_workers'] = self.coco_n_workers

        return config


@dataclass
class DetectorInferConfig(BaseConfig):
    checkpoint_state_dict_path: str

    base_params_config: DetectorBaseParamsConfig
    architecture_config: DetectorArchitectureConfig

    @classmethod
    def from_dict(cls, config: dict):
        detector_infer_config = config['detector']['infer']

        base_params_config = DetectorBaseParamsConfig.from_dict(detector_infer_config['base_params'])
        architecture_config = DetectorArchitectureConfig.from_dict(detector_infer_config['architecture'])

        return cls(
            checkpoint_state_dict_path=detector_infer_config['io']['checkpoint_state_dict_path'],
            base_params_config=base_params_config,
            architecture_config=architecture_config,
        )

    def to_structured_dict(self):
        config = {
            'detector': {
                'infer': {
                    'io': {
                        'checkpoint_state_dict_path': self.checkpoint_state_dict_path,
                    },
                    'base_params': self.base_params_config.to_structured_dict(),
                    'architecture': self.architecture_config.to_structured_dict(),
                }
            }
        }

        return config


@dataclass
class DetectorInferIOConfig(DetectorInferConfig):
    input_tiles_root: str
    infer_aoi_name: str
    output_folder: str
    coco_n_workers: int

    @classmethod
    def from_dict(cls, config: dict):
        parent_config = DetectorInferConfig.from_dict(config)
        detector_infer_io_config = config['detector']['infer']['io']

        return cls(
            **parent_config.as_dict(),
            input_tiles_root=detector_infer_io_config['input_tiles_root'],
            infer_aoi_name=detector_infer_io_config['infer_aoi_name'],
            output_folder=detector_infer_io_config['output_folder'],
            coco_n_workers=detector_infer_io_config['coco_n_workers']
        )

    def to_structured_dict(self):
        config = super().to_structured_dict()
        config['detector']['infer']['io']['input_tiles_root'] = self.input_tiles_root
        config['detector']['infer']['io']['infer_aoi_name'] = self.infer_aoi_name
        config['detector']['infer']['io']['output_folder'] = self.output_folder
        config['detector']['infer']['io']['coco_n_workers'] = self.coco_n_workers

        return config
