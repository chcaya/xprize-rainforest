from dataclasses import dataclass

from config.config_parsers.base_config_parsers import BaseConfig


@dataclass
class EmbedderInferConfig(BaseConfig):
    batch_size: int
    use_pca: bool
    pca_model_path: str
    pca_n_features: int
    pca_n_patches: int

    @classmethod
    def from_dict(cls, config: dict):
        embedder_infer_config = config['embedder']['infer']
        return cls(
            batch_size=embedder_infer_config['batch_size'],
            use_pca=embedder_infer_config['use_pca'],
            pca_model_path=embedder_infer_config['pca_model_path'],
            pca_n_features=embedder_infer_config['pca_n_features'],
            pca_n_patches=embedder_infer_config['pca_n_patches'],
        )

    def to_structured_dict(self) -> dict:
        config = {
            'embedder': {
                'infer': {
                    'batch_size': self.batch_size,
                    'use_pca': self.use_pca,
                    'pca_model_path': self.pca_model_path,
                    'pca_n_features': self.pca_n_features,
                    'pca_n_patches': self.pca_n_patches,
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
