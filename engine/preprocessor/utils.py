import geopandas as gpd
from shapely.geometry import box


def save_package_as_ext(gdf: gpd.GeoDataFrame, output_path: str, output_ext: str or None):
    if output_ext is None:
        # Tries to infer the driver based on output_path file extension
        gdf.to_file(output_path)
    elif output_ext == "gpkg":
        gdf.to_file(output_path, driver="GPKG")
    elif output_ext == "geojson":
        gdf.to_file(output_path, driver="GeoJSON")
    else:
        raise ValueError(f"Wrong value received for output_ext: '{output_ext}'. It should be either 'gpkg' or 'geojson'.")


def masks_to_bounds(source_path: str, output_path: str, output_ext: str = None):
    data_gdf = gpd.read_file(source_path)
    bounding_boxes = data_gdf.geometry.apply(lambda geom: box(*geom.bounds))
    bbox_gdf = gpd.GeoDataFrame(geometry=bounding_boxes)
    save_package_as_ext(gdf=bbox_gdf, output_path=output_path, output_ext=output_ext)


def change_crs_to_epsg_4326(source_path: str, output_path: str, output_ext: str = None):
    # Loading vector packages in rastervision requires that the CRS of the geojson is set to EPSG:4326
    data_gdf = gpd.read_file(source_path)
    data_gdf = data_gdf.to_crs(crs="EPSG:4326")
    save_package_as_ext(gdf=data_gdf, output_path=output_path, output_ext=output_ext)
