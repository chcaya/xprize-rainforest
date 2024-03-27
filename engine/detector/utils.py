import json
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path
import numpy as np
import rasterio
import torch
from geodataset.utils import generate_label_coco
from shapely import box


def collate_fn_detection(batch):
    if type(batch[0][0]) is np.ndarray:
        data = np.array([item[0] for item in batch])
        data = torch.tensor(data, dtype=torch.float32)
    else:
        data = torch.tensor([item[0] for item in batch], dtype=torch.float32)

    labels = [{'boxes': torch.tensor(item[1]['boxes'], dtype=torch.float32),
               'labels': torch.tensor(item[1]['labels'], dtype=torch.long)} for item in batch]

    return data, labels


def collate_fn_images(batch):
    if type(batch[0]) is np.ndarray:
        data = np.array([item for item in batch])
        data = torch.tensor(data, dtype=torch.float32)
    else:
        data = torch.tensor([item for item in batch], dtype=torch.float32)

    return data


def process_tile(tile_data):
    tile_id, (tile_path, tile_boxes, tile_boxes_scores) = tile_data
    local_detections_coco = []

    with rasterio.open(tile_path) as tile:
        tile_width, tile_height = tile.width, tile.height

    for i in range(len(tile_boxes)):
        detection = generate_label_coco(
            polygon=tile_boxes[i],
            tile_height=tile_height,
            tile_width=tile_width,
            tile_id=tile_id,
            use_rle_for_labels=True,
            category_id=None,
            other_attributes_dict={'score': float(tile_boxes_scores[i])}
        )
        local_detections_coco.append(detection)
    return {
        "image_coco": {
            "id": tile_id,
            "width": tile_width,
            "height": tile_height,
            "file_name": str(tile_path.name),
        },
        "detections_coco": local_detections_coco
    }


def generate_detector_inference_coco(raster_name: str,
                                     tiles_paths: list,
                                     boxes: list,
                                     scores: list,
                                     output_path: Path,
                                     n_workers: int):
    images_cocos = []
    detections_cocos = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(process_tile, enumerate(zip(tiles_paths, boxes, scores))))

    for result in results:
        images_cocos.append(result["image_coco"])
        detections_cocos.extend(result["detections_coco"])

    # Save the COCO dataset to a JSON file
    with output_path.open('w') as f:
        json.dump({
            "info": {
                "description": f"Inference for the product '{raster_name}'",
                "dataset_name": raster_name,
                "version": "1.0",
                "year": str(date.today().year),
                "date_created": str(date.today())
            },
            "licenses": [
                # add license?
            ],
            "images": images_cocos,
            "annotations": detections_cocos,
            "categories": None
        }, f, ensure_ascii=False, indent=2)


def detector_result_to_lists(detector_result):
    detector_result = [{k: v.cpu().numpy() for k, v in x.items()} for x in detector_result]
    for x in detector_result:
        x['boxes'] = [box(*b) for b in x['boxes']]
        x['scores'] = x['scores'].tolist()
    boxes = [x['boxes'] for x in detector_result]
    scores = [x['scores'] for x in detector_result]

    return boxes, scores
