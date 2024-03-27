from pathlib import Path


from geodataset.tilerize import RasterTilerizer, LabeledRasterTilerizer
from geodataset.aoi import AOIGeneratorConfig, AOIFromPackageConfig

from config.config_parsers.tilerizer_parsers import TilerizerCLIConfig


def tilerizer_main(config: TilerizerCLIConfig):
    output_folder = Path(config.output_folder)
    output_folder.mkdir(exist_ok=False, parents=True)

    if not config.aoi_config:
        aois_config = AOIGeneratorConfig(
            aoi_type="band",
            aois={'all': {'percentage': 100, 'position': 1}}
        )
    elif config.aoi_config == "generate":
        aois_config = AOIGeneratorConfig(
            aoi_type=config.aoi_type,
            aois=config.aois
        )
    elif config.aoi_config == "package":
        aois_config = AOIFromPackageConfig(
            aois=config.aois
        )
    else:
        raise ValueError(f"Unsupported value for aoi_config {config.aoi_config}.")

    if config.labels_path:
        tilerizer = LabeledRasterTilerizer(
            raster_path=Path(config.raster_path),
            labels_path=Path(config.labels_path),
            output_path=Path(config.output_folder),
            tile_size=config.tile_size,
            tile_overlap=config.tile_overlap,
            aois_config=aois_config,
            scale_factor=config.scale_factor,
            ground_resolution=config.ground_resolution,
            ignore_black_white_alpha_tiles_threshold=config.ignore_black_white_alpha_tiles_threshold,
            ignore_tiles_without_labels=config.ignore_tiles_without_labels,
            main_label_category_column_name=config.main_label_category_column_name)

        tilerizer.generate_coco_dataset()
    else:
        tilerizer = RasterTilerizer(
            raster_path=Path(config.raster_path),
            output_path=Path(config.output_folder),
            tile_size=config.tile_size,
            tile_overlap=config.tile_overlap,
            aois_config=aois_config,
            scale_factor=config.scale_factor,
            ground_resolution=config.ground_resolution,
            ignore_black_white_alpha_tiles_threshold=config.ignore_black_white_alpha_tiles_threshold,
        )

        tilerizer.generate_tiles()

    config.save_yaml_config(output_path=output_folder / "tilerizer_config.yaml")

    return tilerizer.tiles_path


