from pathlib import Path

import geopandas as gpd
from datetime import datetime
import time

import numpy as np


def align_labels_to_polygons(labels, segmentations, stride=1.0, max_dist=8):
    best_x_off = None
    best_y_off = None
    best_total_intersection_area = 0

    labels.geometry = labels.buffer(0)
    segmentations.geometry = segmentations.buffer(0)

    start_time = time.time()
    for x_off in np.arange(-max_dist, max_dist + stride, stride):
        for y_off in np.arange(-max_dist, max_dist + stride, stride):
            labels_copy = labels.copy(deep=True)
            segmentations_copy = segmentations.copy(deep=True)
            labels_copy['geometry'] = labels_copy['geometry'].translate(xoff=x_off, yoff=y_off)

            # compute the intersection between the labels and the segmentations
            intersection = gpd.overlay(labels_copy, segmentations_copy, how='intersection', keep_geom_type=False)

            # compute the area of the intersection
            intersection['intersection_area'] = intersection.area

            # get the total intersection area
            total_intersection_area = intersection['intersection_area'].sum()

            if total_intersection_area > best_total_intersection_area:
                best_total_intersection_area = total_intersection_area
                best_x_off = x_off
                best_y_off = y_off

    print(f'best_x_off: {best_x_off},'
          f' best_y_off: {best_y_off},'
          f' best_total_intersection_area: {best_total_intersection_area},'
          f' latency: {time.time() - start_time} seconds.\n')

    aligned_labels = labels.copy(deep=True)
    aligned_labels['geometry'] = aligned_labels['geometry'].translate(xoff=best_x_off, yoff=best_y_off)

    return aligned_labels


if __name__ == '__main__':
    output_folder = '/media/hugobaudchon/4 TB/XPrize/Data/panama_BCI_aligned_labels'
    sam_polygons_folder = '/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations/panama'

    labels_2020_path = '/media/hugobaudchon/4 TB/XPrize/Data/raw/panama/20200801_bci50ha_p4pro/20200801_bci50ha_p4pro_labels_masks.gpkg'
    labels_2022_path = '/media/hugobaudchon/4 TB/XPrize/Data/raw/panama/20220929_bci50ha_p4pro/20220929_bci50ha_p4pro_labels_masks.gpkg'

    labels_2020 = gpd.read_file(labels_2020_path)
    labels_2022 = gpd.read_file(labels_2022_path)

    labels_2020_date = datetime.strptime('2020-08-01', '%Y-%m-%d')
    labels_2022_date = datetime.strptime('2022-09-29', '%Y-%m-%d')

    for segmentations_path in list(Path(sam_polygons_folder).rglob('*.gpkg')):
        if "local" not in segmentations_path.stem:
            continue

        segmentations = gpd.read_file(str(segmentations_path))

        # parse segmentation date from path
        segmentations_date = datetime.strptime("-".join(str(segmentations_path).split('/')[-1].split('_')[2:5]), '%Y-%m-%d')

        # find out the closest date (from labels_2020_date and labels_2022_date) to the segmentations_date
        print(segmentations_date, abs(labels_2020_date - segmentations_date), abs(labels_2022_date - segmentations_date))
        if abs(labels_2020_date - segmentations_date) < abs(labels_2022_date - segmentations_date):
            print('using 2020 labels')
            labels = labels_2020
            labels_date_str = '2020-08-01'
        else:
            print('using 2022 labels')
            labels = labels_2022
            labels_date_str = '2022-09-29'

        output_file = Path(output_folder) / f"{segmentations_path.stem}_label_{labels_date_str}_aligned.gpkg"

        if output_file.exists():
            continue

        aligned_labels = align_labels_to_polygons(labels, segmentations, stride=0.50, max_dist=3)

        aligned_labels.to_file(output_file, driver='GPKG')
