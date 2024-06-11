from pathlib import Path

from geodataset.utils import CocoNameConvention, GeoPackageNameConvention, coco_to_geopackage


from config.config_parsers.coco_to_geopackage_parsers import CocoToGeopackageIOConfig


def coco_to_geopackage_main(config: CocoToGeopackageIOConfig):
    output_folder = Path(config.output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)

    print('Converting COCO file to geopackage...')

    product_name, scale_factor, ground_resolution, fold = CocoNameConvention.parse_name(Path(config.coco_path).name)

    geojson_name = GeoPackageNameConvention.create_name(product_name=product_name,
                                                        fold=fold,
                                                        scale_factor=scale_factor,
                                                        ground_resolution=ground_resolution)

    geojson_output_path = output_folder / geojson_name

    gdf = coco_to_geopackage(
        coco_json_path=config.coco_path,
        images_directory=config.input_tiles_root,
        convert_to_crs_coordinates=True,
        geopackage_output_path=str(geojson_output_path),
    )

    return gdf, geojson_output_path
