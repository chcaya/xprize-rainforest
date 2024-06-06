from dataclasses import dataclass

from config.config_parsers.base_config_parsers import BaseConfig


@dataclass
class CocoToGeopackageIOConfig(BaseConfig):
    input_tiles_root: str
    coco_path: str
    output_folder: str

    @classmethod
    def from_dict(cls, config: dict):
        coco_to_geopackage_io_config = config['coco_to_geopackage']['io']
        return cls(
            input_tiles_root=coco_to_geopackage_io_config['input_tiles_root'],
            coco_path=coco_to_geopackage_io_config['coco_path'],
            output_folder=coco_to_geopackage_io_config['output_folder'],
        )

    def to_structured_dict(self):
        config = {
            'coco_to_geopackage': {
                'io': {
                    'input_tiles_root': self.input_tiles_root,
                    'coco_path': self.coco_path,
                    'output_folder': self.output_folder,
                }
            }
        }

        return config
