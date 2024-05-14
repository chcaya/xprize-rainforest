from pathlib import Path
from typing import List

import numpy as np
import torch
from geodataset.dataset import BoxesDataset, DetectionLabeledRasterCocoDataset
from geodataset.utils import apply_affine_transform, COCOGenerator
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

    def _infer(self, image: np.ndarray, boxes: List[np.array]):
        self.predictor.set_image(image)
        box_array = np.array(boxes)
        box_tensor = torch.Tensor(box_array).to(self.device).to(torch.long)

        masks, scores, low_res_masks = self.predictor.predict_torch(point_coords=None,
                                                                    point_labels=None,
                                                                    boxes=box_tensor,
                                                                    multimask_output=False)

        return masks, scores

    def infer_on_single_box_dataset(self, boxes_dataset: BoxesDataset, geojson_output_path: str):
        mask_polygons = []
        dataset_with_progress = tqdm(boxes_dataset,
                                     desc="Inferring SAM...",
                                     leave=True)
        for image, box, (minx, miny) in dataset_with_progress:
            image = image[:3, :, :]
            image_hwc = image.transpose((1, 2, 0))
            masks = self._infer(image=image_hwc, boxes=[box.bounds])

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

    def infer_on_multi_box_dataset(self, dataset: DetectionLabeledRasterCocoDataset, coco_json_output_path: Path):
        dataset_with_progress = tqdm(dataset,
                                     desc="Inferring SAM...",
                                     leave=True)
        tiles_paths = []
        tiles_masks = []
        tiles_scores = []
        for tile_idx, (image, boxes_data) in enumerate(dataset_with_progress):
            image = image[:3, :, :]
            image_hwc = image.transpose((1, 2, 0))
            image_hwc = (image_hwc * 255).astype(np.uint8)
            masks, scores = self._infer(image=image_hwc, boxes=boxes_data['boxes'])
            masks = masks.cpu().numpy()
            scores = scores.cpu().numpy()
            masks_polygons = [mask_to_polygon(mask.squeeze(), simplify_tolerance=self.simplify_tolerance) for mask
                              in masks]
            adjusted_masks_polygons = [translate(mask_polygon, xoff=0, yoff=0) for mask_polygon in masks_polygons]

            tiles_paths.append(dataset.tiles[tile_idx]['path'])
            tiles_masks.append(adjusted_masks_polygons)

            scores_pp = scores.squeeze().tolist()
            if isinstance(scores_pp, float):
                scores_pp = [scores_pp]
            tiles_scores.append(scores_pp)

        coco_generator = COCOGenerator(
            description=f"Aggregated boxes from multiple tiles.",
            tiles_paths=tiles_paths,
            polygons=tiles_masks,
            scores=tiles_scores,
            categories=None,  # TODO add support for categories
            other_attributes=None,  # TODO add support for other_attributes
            output_path=coco_json_output_path,
            use_rle_for_labels=True,  # TODO make this a parameter to the class
            n_workers=5,  # TODO make this a parameter to the class
            coco_categories_list=None  # TODO make this a parameter to the class
        )
        coco_generator.generate_coco()
