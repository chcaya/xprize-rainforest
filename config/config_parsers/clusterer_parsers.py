from dataclasses import dataclass

from config.config_parsers.base_config_parsers import BaseConfig


@dataclass
class ClustererInferConfig(BaseConfig):
    clustering_algorithm: str
    clustering_algorithm_params: dict

    @classmethod
    def from_dict(cls, config: dict):
        clusterer_config = config['clusterer']

        return cls(
            clustering_algorithm=clusterer_config['clustering_algorithm'],
            clustering_algorithm_params=clusterer_config['clustering_algorithm_params']
        )

    def to_structured_dict(self) -> dict:
        config = {
            'clusterer': {
                'clustering_algorithm': self.clustering_algorithm,
                'clustering_algorithm_params': self.clustering_algorithm_params
            }
        }

        return config



@dataclass
class ClustererInferIOConfig(ClustererInferConfig):
    input_dataframe: str
    output_folder: str

    @classmethod
    def from_dict(cls, config: dict):
        clusterer_config = config['clusterer']
        clusterer_io_config = config['clusterer']['io']

        return cls(
            **clusterer_config,
            input_dataframe=clusterer_io_config['input_dataframe'],
            output_folder=clusterer_io_config['output_folder'],
        )

    def to_structured_dict(self):
        config = {
            'embedder': {
                'io': {
                    'input_tiles_root': self.input_tiles_root,
                    'coco_path': self.coco_path,
                    'output_folder': self.output_folder
                }
            }
        }

        return config
