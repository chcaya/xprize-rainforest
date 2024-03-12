import itertools
import math
from typing import List, Tuple

import numpy as np
import geopandas as gpd
import pandas as pd
from shapely import box
from shapely.affinity import translate


class Aggregator:
    def __init__(self,
                 tiles_coordinates: List[Tuple[int, int]],
                 tiles_boxes: List[List[box]],
                 tiles_boxes_scores: List[list],
                 tile_overlap: float,
                 tile_size: int):
        self.tiles_coordinates = tiles_coordinates
        self.tiles_boxes = tiles_boxes
        self.tiles_boxes_scores = tiles_boxes_scores
        self.tile_overlap = tile_overlap
        self.tile_size = tile_size

        self.tiles_to_boxes_map = {k: v for k, v in zip(self.tiles_coordinates, self.tiles_boxes)}
        self.tiles_to_boxes_scores_map = {k: v for k, v in zip(self.tiles_coordinates, self.tiles_boxes_scores)}
        self.adjacent_tiles_map = self.find_adjacent_tiles()

    def find_adjacent_tiles(self):
        tiles_spacing = 1 - self.tile_overlap
        overlapping_radius = math.ceil(round((0.5 / tiles_spacing), 10))  # Rounding to 10th decimal to take care of floating point precision errors. For example, when overlap=0.9, then tiles_spacing=0.09999999999999998 (should be 0.1) and then 0.5 / tiles_spacing = 5.000000000000001 (should be 5) and then ceil(0.5 / tiles_spacing) = 6 (should be 5).
        adjacent_tiles_map = {k: [] for k in self.tiles_coordinates}
        for tile_coordinates in self.tiles_coordinates:
            c_x = tile_coordinates[0]
            c_y = tile_coordinates[1]
            for r_x in range(-overlapping_radius, overlapping_radius + 1):
                for r_y in range(-overlapping_radius, overlapping_radius + 1):
                    if r_x == 0 and r_y == 0:
                        continue
                    if (c_x + r_x, c_y + r_y) in adjacent_tiles_map:
                        adjacent_tiles_map[(c_x, c_y)].append((c_x + r_x, c_y + r_y))

        adjacent_tiles_map = {k: set(v) for k, v in adjacent_tiles_map.items()}
        return adjacent_tiles_map

    def aggregate_boxes(self):
        aggregated_boxes = {k: [] for k in self.adjacent_tiles_map}
        for tile_coordinates in self.adjacent_tiles_map:
            tile_boxes = self.tiles_to_boxes_map[tile_coordinates]
            tile_boxes_scores = self.tiles_to_boxes_scores_map[tile_coordinates]
            adjacent_tiles_coordinates = self.adjacent_tiles_map[tile_coordinates]
            adjacent_tiles_boxes = [self.tiles_to_boxes_map[coords] for coords in adjacent_tiles_coordinates]
            adjacent_tiles_boxes_scores = [self.tiles_to_boxes_scores_map[coords] for coords in adjacent_tiles_coordinates]

            tile_gdf = gpd.GeoDataFrame({'tile_id': [tile_coordinates] * len(tile_boxes),
                                         'geometry': tile_boxes,
                                         'scores': tile_boxes_scores})

            # Making sure all the boxes of the overlapping tiles are in the same coordinate system as the center tile.
            adjacent_tiles_gdfs = []
            for a_tile_coordinates, a_tile_boxes, a_tile_boxes_scores in zip(adjacent_tiles_coordinates,
                                                                             adjacent_tiles_boxes,
                                                                             adjacent_tiles_boxes_scores):
                a_tiles_gdf = gpd.GeoDataFrame({'tile_id': [a_tile_coordinates] * len(a_tile_boxes),
                                                'geometry': a_tile_boxes,
                                                'scores': a_tile_boxes_scores})
                x_shift = (a_tile_coordinates[0] - tile_coordinates[0]) * self.tile_size
                y_shift = (a_tile_coordinates[1] - tile_coordinates[1]) * self.tile_size
                a_tiles_gdf['geometry'] = a_tiles_gdf['geometry'].astype(object).apply(
                    lambda geom: translate(geom, xoff=x_shift, yoff=y_shift)
                )
                adjacent_tiles_gdfs.append(a_tiles_gdf)

            adjacent_tiles_gdf = gpd.GeoDataFrame(pd.concat(adjacent_tiles_gdfs, ignore_index=True))

            # Computing the IoU between boxes
            intersections = gpd.overlay(tile_gdf, adjacent_tiles_gdf, how='intersection')

            tile_gdf['area'] = tile_gdf.geometry.area
            adjacent_tiles_gdf['area'] = adjacent_tiles_gdf.geometry.area
            intersections['intersection_area'] = intersections.geometry.area

            intersections['iou'] = intersections['intersection_area'] / (tile_gdf['area'].iloc[0] + adjacent_tiles_gdf['area'].iloc[0] - intersections['intersection_area'])
            print(tile_gdf)
            print(adjacent_tiles_gdf)
            print(intersections)


if __name__ == "__main__":
    tiles_coordinates = [(x, y) for x in range(5) for y in range(4)]  # Example coordinates
    tiles_boxes = [[box(0, 0, 1, 1), box(0, 0, 1, 1)] for _ in range(20)]  # Placeholder boxes for simplicity
    tiles_boxes_scores = [[0.9, 0.7] for _ in range(20)]  # Placeholder scores for simplicity

    aggregator = Aggregator(tiles_coordinates, tiles_boxes, tiles_boxes_scores, tile_overlap=0.5, tile_size=1024)
    for x in aggregator.adjacent_tiles_map:
        print(x, aggregator.adjacent_tiles_map[x])
    aggregator.aggregate_boxes()