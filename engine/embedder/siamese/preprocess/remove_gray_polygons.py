import time
from pathlib import Path

import numpy as np
from geodataset.geodata import Raster
from geodataset.labels import RasterPolygonLabels
from geodataset.utils import apply_affine_transform


def remove_gray_objects(output_folder: Path, raster: Raster, polygons_path: Path, gray_ratio: float, tolerance: int, lower_threshold: int):
    n_start = len(geometries_gdf)
    t_start = time.time()
    for i, polygon_row in geometries_gdf.iterrows():
        # get the pixels within the polygon
        polygon = polygon_row['geometry']

        # get max width and height of the polygon
        minx, miny, maxx, maxy = polygon.bounds
        width = maxx - minx
        height = maxy - miny

        polygon_tile, translated_polygon_intersection = raster.get_polygon_tile(polygon, polygon_id=i, tile_size=int(max(width, height)) + 50)

        pixels = polygon_tile.data

        red_band = pixels[0, :, :]
        green_band = pixels[1, :, :]
        blue_band = pixels[2, :, :]

        # Create a mask for gray pixels within the defined tolerance and above the lower threshold
        gray_mask = (
                    (np.abs(red_band - green_band) < tolerance) &
                    (np.abs(red_band - blue_band) < tolerance) &
                    (np.abs(green_band - blue_band) < tolerance) &
                    (red_band >= lower_threshold) &
                    (green_band >= lower_threshold) &
                    (blue_band >= lower_threshold))

        # Count the number of gray pixels
        num_gray_pixels = np.sum(gray_mask)

        if num_gray_pixels / int(polygon.area) > gray_ratio:
            # rgba_image = np.transpose(pixels, (1, 2, 0))
            # fig, ax = plt.subplots(figsize=(10, 10))  # Adjust figure size as needed
            # ax.imshow(rgba_image)
            # ax.set_title('Zoomed-in RGBA Image')
            # ax.axis('off')  # Hide the axis  # Hide the axis
            # plt.show()

            geometries_gdf.drop(i, inplace=True)

    geometries_gdf['geometry'] = geometries_gdf.apply(lambda row: apply_affine_transform(row['geometry'], raster.metadata['transform']), axis=1)
    geometries_gdf.crs = raster.metadata['crs']

    geometries_gdf.to_file(str(output_folder / f"{polygons_path.stem}_no_gray_{str(gray_ratio).replace('.', 'p')}_{tolerance}_{lower_threshold}.gpkg"), driver='GPKG')

    print(f"Raster {raster.name}: Removed {n_start - len(geometries_gdf)}/{n_start} polygons with more than {gray_ratio * 100}% gray pixels"
          f" in {time.time() - t_start} seconds.")


if __name__ == '__main__':
    gray_ratio = 0.0012  # 0.0008
    tolerance = 10
    lower_threshold = 200

    output_folder = Path('/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations/panama_no_gray_2')

    # rasters_paths = [Path('/media/hugobaudchon/4 TB/XPrize/Data/raw/panama/BCI_50ha_timeseries_local_alignment/BCI_50ha_2018_04_04_local.tif')]
    # polygons_paths = [Path('/media/hugobaudchon/4 TB/XPrize/infer/FINAL_SAM_dataset/BCI_50ha_2018_04_04_local/segmenter_aggregator_output/bci_50ha_2018_04_04_local_gr0p08_infersegmenteraggregator.gpkg')]

    rasters_paths = list(Path('/media/hugobaudchon/4 TB/XPrize/Data/raw/panama/BCI_50ha_timeseries_local_alignment').rglob('*.tif'))
    all_polygons_paths = list(Path('/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations/panama').rglob('*.gpkg'))

    polygons_paths = []
    for raster_path in rasters_paths:
        for polygon_path in all_polygons_paths:
            if raster_path.stem.lower() in polygon_path.stem:
                polygons_paths.append(polygon_path)

    for raster_path, polygons_path in zip(rasters_paths, polygons_paths):
        raster = Raster(path=raster_path)
        polygons = RasterPolygonLabels(path=polygons_path,
                                       associated_raster=raster)

        geometries_gdf = polygons.geometries_gdf

        remove_gray_objects(
            output_folder,
            raster,
            polygons_path,
            gray_ratio,
            tolerance,
            lower_threshold
        )






