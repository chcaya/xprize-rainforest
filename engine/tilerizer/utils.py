from geodataset.aoi import AOIGeneratorConfig, AOIFromPackageConfig

from config.config_parsers.tilerizer_parsers import TilerizerConfig


def parse_tilerizer_aoi_config(config: TilerizerConfig):
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

    return aois_config
