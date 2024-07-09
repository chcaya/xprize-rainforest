from pathlib import Path
from typing import List

import albumentations
import numpy as np
import rasterio
from geodataset.dataset.base_dataset import BaseLabeledRasterCocoDataset
from geodataset.utils import rle_segmentation_to_mask, mask_to_polygon


class DINOv2SegmentationLabeledRasterCocoDataset(BaseLabeledRasterCocoDataset):
    def __init__(self, fold: str, root_path: Path or List[Path],
                 transform: albumentations.core.composition.Compose = None,
                 image_size_center_crop_pad: int = None):
        super().__init__(fold=fold, root_path=root_path, transform=transform)

        self.image_size_center_crop_pad = image_size_center_crop_pad

    def __getitem__(self, idx: int):
        """
        Retrieves a tile and its segmentations/masks by index, applying any specified transformations.

        Parameters:
        - idx: int, the index of the tile to retrieve.

        Returns:
        - A tuple containing the transformed tile and its segmentations/masks.
        """
        tile_info = self.tiles[idx]

        with rasterio.open(tile_info['path']) as tile_file:
            tile = tile_file.read([1, 2, 3])  # Reading the first three bands

        labels = tile_info['labels']
        masks = []
        polygons = []
        for label in labels:
            if 'segmentation' in label:
                segmentation = label['segmentation']
                if ('is_rle_format' in label and label['is_rle_format']) or isinstance(segmentation, dict):
                    # RLE format
                    mask = rle_segmentation_to_mask(segmentation)
                    polygon = mask_to_polygon(mask)
                    polygons.append(polygon)
                    masks.append(mask)
                else:
                    raise NotImplementedError("Please make sure that the masks are encoded using RLE.")

        category_ids = np.array([0 if label['category_id'] is None else label['category_id']
                                 for label in labels])

        if self.transform:
            transformed = self.transform(image=tile.transpose((1, 2, 0)),
                                         mask=np.stack(masks, axis=0),
                                         labels=category_ids)
            transformed_image = transformed['image'].transpose((2, 0, 1))
            transformed_masks = [mask for mask in transformed['mask']]
            transformed_category_ids = transformed['labels']
        else:
            transformed_image = tile
            transformed_masks = masks
            transformed_category_ids = category_ids

        if self.image_size_center_crop_pad:
            image_size = self.image_size_center_crop_pad
            if transformed_image.shape[1] > image_size:
                # crop
                data_center = int(transformed_image.shape[1] / 2)
                data = transformed_image[:,
                       data_center - image_size // 2:data_center + image_size // 2,
                       data_center - image_size // 2:data_center + image_size // 2,
                       ]
                masks = [mask[
                         data_center - image_size // 2:data_center + image_size // 2,
                         data_center - image_size // 2:data_center + image_size // 2,
                         ] for mask in masks]
                transformed_image = data
                transformed_masks = masks
            elif transformed_image.shape[1] < image_size:
                # pad
                padding = (image_size - transformed_image.shape[1]) // 2
                padded_data = np.zeros((transformed_image.shape[0], image_size, image_size),
                                       dtype=transformed_image.dtype)
                padded_data[:, padding:padding + transformed_image.shape[1], padding:padding + transformed_image.shape[1]] = transformed_image
                data = padded_data
                masks = []
                for transformed_mask in transformed_masks:
                    padded_mask = np.zeros((image_size, image_size), dtype=transformed_mask.dtype)
                    padded_mask[padding:padding + transformed_mask.shape[0], padding:padding + transformed_mask.shape[0]] = transformed_mask
                    masks.append(padded_mask)
                transformed_image = data
                transformed_masks = masks

        transformed_image = transformed_image / 255  # normalizing
        area = np.array([np.sum(mask) for mask in masks])
        # suppose all instances are not crowd
        iscrowd = np.zeros((len(transformed_masks),))
        # get tile id
        image_id = np.array([idx])
        transformed_masks = {'masks': transformed_masks, 'labels': transformed_category_ids,
                             'area': area, 'iscrowd': iscrowd, 'image_id': image_id, 'labels_polygons': polygons}

        return transformed_image, transformed_masks
