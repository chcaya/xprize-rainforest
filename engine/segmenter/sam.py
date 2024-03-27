from pathlib import Path

import numpy as np
import shapely
import torch
from geodataset.dataset import BoxesDataset
from geodataset.utils import apply_affine_transform
from geopandas import GeoDataFrame
from segment_anything import SamPredictor, sam_model_registry
from shapely.affinity import translate
from tqdm import tqdm

from engine.segmenter.utils import mask_to_polygon


class SamPredictorWrapper:
    def __init__(self,
                 model_type: str,
                 checkpoint_path: str,
                 simplify_tolerance: float):
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.simplify_tolerance = simplify_tolerance
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
        sam.to(self.device)
        self.predictor = SamPredictor(sam)

    def infer(self, image: np.ndarray, box: shapely.box):
        self.predictor.set_image(image)
        box_array = np.array(box.bounds)
        masks, _, _ = self.predictor.predict(box=box_array, multimask_output=False)
        return masks

    def infer_on_dataset(self, boxes_dataset: BoxesDataset, geojson_output_path: str):
        mask_polygons = []
        dataset_with_progress = tqdm(boxes_dataset,
                                     desc="Inferring SAM...",
                                     leave=True)
        for image, box, (minx, miny) in dataset_with_progress:
            image = image[:3, :, :]
            image_hwc = image.transpose((1, 2, 0))
            masks = self.infer(image=image_hwc, box=box)

            mask_polygon = mask_to_polygon(masks.squeeze(), simplify_tolerance=self.simplify_tolerance)

            adjusted_mask_polygon = translate(mask_polygon, xoff=minx, yoff=miny)
            mask_polygons.append(adjusted_mask_polygon)

        gdf = GeoDataFrame(geometry=mask_polygons)
        gdf['geometry'] = gdf['geometry'].astype(object).apply(
            lambda geom: apply_affine_transform(geom, boxes_dataset.raster.metadata['transform'])
        )
        gdf.set_crs(boxes_dataset.raster.metadata['crs'], inplace=True)
        gdf.to_file(geojson_output_path, driver='GeoJSON')

        return gdf

