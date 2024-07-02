from dataclasses import dataclass

from config.config_parsers.base_config_parsers import BaseConfig, BaseIntermediateConfig


@dataclass
class EmbedderArchitectureConfig(BaseIntermediateConfig):
    architecture_name: str
    backbone_model_resnet_name: str
    final_embedding_size: int

    @classmethod
    def from_dict(cls, config: dict):
        return cls(**config)

    def to_structured_dict(self):
        config = {
            'architecture_name': self.architecture_name,
            'backbone_model_resnet_name': self.backbone_model_resnet_name,
            'final_embedding_size': self.final_embedding_size,
        }

        return config


@dataclass
class EmbedderInferConfig(BaseConfig):
    batch_size: int

    @classmethod
    def from_dict(cls, config: dict):
        embedder_infer_config = config['embedder']['infer']
        return cls(
            batch_size=embedder_infer_config['batch_size'],
        )

    def to_structured_dict(self) -> dict:
        config = {
            'embedder': {
                'infer': {
                    'batch_size': self.batch_size,
                }
            }
        }

        return config


@dataclass
class EmbedderInferIOConfig(BaseConfig):
    input_tiles_root: str
    coco_path: str
    output_folder: str

    @classmethod
    def from_dict(cls, config: dict):
        embedder_infer_io_config = config['embedder']['infer']['io']

        return cls(
            input_tiles_root=embedder_infer_io_config['input_tiles_root'],
            coco_path=embedder_infer_io_config['coco_path'],
            output_folder=embedder_infer_io_config['output_folder'],
        )

    def to_structured_dict(self):
        config = {
            'embedder': {
                'infer': {
                    'io': {
                        'input_tiles_root': self.input_tiles_root,
                        'coco_path': self.coco_path,
                        'output_folder': self.output_folder
                    }
                }
            }
        }

        return config


@dataclass
class SiameseInferConfig(EmbedderInferConfig):
    checkpoint_path: str
    architecture_config: EmbedderArchitectureConfig

    @classmethod
    def from_dict(cls, config: dict):
        parent_config = EmbedderInferConfig.from_dict(config)
        embedder_infer_config = config['embedder']['infer']
        siamese_config = embedder_infer_config['siamese']

        return cls(
            **parent_config.as_dict(),
            checkpoint_path=siamese_config['checkpoint_path'],
            architecture_config=EmbedderArchitectureConfig.from_dict(siamese_config['architecture']),
        )

    def to_structured_dict(self) -> dict:
        config = super().to_structured_dict()
        config['embedder']['infer']['siamese'] = {
            'checkpoint_path': self.checkpoint_path,
            'architecture': self.architecture_config.to_structured_dict()
        }

        return config


@dataclass
class SiameseInferIOConfig(SiameseInferConfig, EmbedderInferIOConfig):
    @classmethod
    def from_dict(cls, config: dict):
        embedder_infer_io_config = EmbedderInferIOConfig.from_dict(config)
        siamese_config = SiameseInferConfig.from_dict(config)

        return cls(
            **embedder_infer_io_config.as_dict(),
            **siamese_config.as_dict(),
        )

    def to_structured_dict(self):
        embedder_infer_io_config = EmbedderInferIOConfig.to_structured_dict(self)
        siamese_config = SiameseInferConfig.to_structured_dict(self)
        siamese_config['embedder']['infer']['io'] = embedder_infer_io_config['embedder']['infer']['io']
        return siamese_config


@dataclass
class ContrastiveInferConfig(EmbedderInferConfig):
    checkpoint_path: str
    mean_std_descriptor: str
    image_size: int

    @classmethod
    def from_dict(cls, config: dict):
        parent_config = EmbedderInferConfig.from_dict(config)
        embedder_infer_config = config['embedder']['infer']
        contrastive_config = embedder_infer_config['contrastive']

        return cls(
            **parent_config.as_dict(),
            checkpoint_path=contrastive_config['checkpoint_path'],
            mean_std_descriptor=contrastive_config['mean_std_descriptor'],
            image_size=contrastive_config['image_size']
        )

    def to_structured_dict(self) -> dict:
        config = super().to_structured_dict()
        config['embedder']['infer']['contrastive'] = {
            'checkpoint_path': self.checkpoint_path,
            'mean_std_descriptor': self.mean_std_descriptor,
            'image_size': self.image_size
        }

        return config


@dataclass
class DINOv2InferConfig(EmbedderInferConfig):
    size: str

    @classmethod
    def from_dict(cls, config: dict):
        parent_config = EmbedderInferConfig.from_dict(config)
        embedder_infer_config = config['embedder']['infer']
        dino_v2_config = embedder_infer_config['dino_v2']

        return cls(
            **parent_config.as_dict(),
            size=dino_v2_config['size'],
        )

    def to_structured_dict(self):
        config = super().to_structured_dict()
        config['embedder']['infer']['dino_v2'] = {
            'size': self.size
        }

        return config


@dataclass
class DINOv2InferIOConfig(DINOv2InferConfig, EmbedderInferIOConfig):
    @classmethod
    def from_dict(cls, config: dict):
        embedder_infer_io_config = EmbedderInferIOConfig.from_dict(config)
        dinov2_config = DINOv2InferConfig.from_dict(config)

        return cls(
            **embedder_infer_io_config.as_dict(),
            **dinov2_config.as_dict(),
        )

    def to_structured_dict(self):
        embedder_infer_io_config = EmbedderInferIOConfig.to_structured_dict(self)
        dino_v2_config = DINOv2InferConfig.to_structured_dict(self)
        dino_v2_config['embedder']['infer']['io'] = embedder_infer_io_config['embedder']['infer']['io']
        return dino_v2_config
